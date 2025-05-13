import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# REPO_BASE_DIR = "/home/zifengw2/DSAgent/DSWizard"
REPO_BASE_DIR = "/Users/zifeng/Documents/github/DSWizard"

# extract the agent type, which is composed of the agent_type, and it's hyperparameters
def _get_agent_type(row) -> str:
    """
    Format agent type based on configuration.
    For react agent: (react, step_count)
    For reasoning coder: (reasoning_coder, planning_model, coding_model)
    For coder: (coder, model_name)
    """
    agent_type = row["agent_type"]
    if agent_type == "react" and row['model_name'] == "o3-mini":
        return "reasoning_react"
    elif agent_type == "react" and row['model_name'] == "gpt-4o":
        return "react"
    elif agent_type == "reasoning_coder":
        return f"reasoning_coder"
    elif agent_type == "coder":
        return f"coder, {row['model_name']}"
    elif agent_type == "reasoning_react":
        return f"reasoning_react, {row['plan_model_name']}"
    return agent_type

def get_experiment_results(logs_directory_path: str):
    """
    Get the results of all experiments in the given directory.
    """
    results = []
    for log_file in os.listdir(logs_directory_path):
        if not log_file.endswith(".json") or "experiment|" not in log_file:
            continue
        with open(os.path.join(logs_directory_path, log_file), "r") as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {log_file}")
    df_results = pd.DataFrame(results)
    df_results = pd.concat([df_results.drop('experiment_config', axis=1), df_results['experiment_config'].apply(pd.Series)], axis=1)
    df_results = df_results[df_results['agent_type'] != "reasoning_react"].reset_index(drop=True)
    df_results['agent_type'] = df_results.apply(_get_agent_type, axis=1)
    return df_results


def load_analysis_classification():
    output_dir = os.path.join(REPO_BASE_DIR, "benchmark_datasets/cBioPortal/classification")
    df = pd.read_csv(os.path.join(output_dir, "analysis_classification.csv"))
    return df


def load_publication_classification():
    output_dir = os.path.join(REPO_BASE_DIR, "benchmark_datasets/cBioPortal/classification")
    df = pd.read_csv(os.path.join(output_dir, "publication_classification.csv"))
    return df


def plot_type_i_ii_error_per_pub_type():
    logs_directory_path = os.path.join(REPO_BASE_DIR, "logs/logs_short_complete")
    results = get_experiment_results(logs_directory_path)
    df_publication = load_publication_classification()
    df_publication['PMID'] = df_publication['PMID'].astype(int).astype(str)
    # convert to PMID to class map
    pmid_to_class = dict(zip(df_publication['PMID'], df_publication['class']))
    results['pub_type'] = results['pmid'].map(pmid_to_class)

    results['clean_final_answer'] = results['final_answer'].str.lower().apply(lambda x: "true" if x.startswith("true") else "false" if x.startswith("false") else x)
    results['clean_final_answer'] = results['clean_final_answer'].str.strip()

    # for each agent type, count the final answer
    for agent_type in results['agent_type'].unique():
        print("-"*100)
        print(agent_type)
        print(results[results['agent_type'] == agent_type]['clean_final_answer'].value_counts())
        
    # plot type I error per pub type
    # type I error: false positive
    # for each pub type, and agent type, count the number of samples which hypothesis is false and final answer is true
    fp_count = results[(results['hypothesis_is_true'] == False) & (results['clean_final_answer'] == "true")].groupby(['pub_type', 'agent_type']).size().reset_index(name='fp_count')
    # for each pub type, and agent type, count the total number of samples
    total_count = results.groupby(['pub_type', 'agent_type']).size().reset_index(name='total_count')
    # merge the two dataframes
    fp_count = fp_count.merge(total_count, on=['pub_type', 'agent_type'], how='left')
    # calculate the false positive rate
    # also calculate the 95% confidence interval
    fp_count['fp_rate'] = fp_count['fp_count'] / fp_count['total_count']
    fp_count['ci_lower'] = fp_count['fp_rate'] - 1.96 * np.sqrt((fp_count['fp_rate'] * (1 - fp_count['fp_rate'])) / fp_count['total_count'])
    fp_count['ci_upper'] = fp_count['fp_rate'] + 1.96 * np.sqrt((fp_count['fp_rate'] * (1 - fp_count['fp_rate'])) / fp_count['total_count'])

    # type II error: false negative
    # for each pub type, and agent type, count the number of samples which hypothesis is true and final answer is false
    fn_count = results[(results['hypothesis_is_true'] == True) & (results['clean_final_answer'] == "false")].groupby(['pub_type', 'agent_type']).size().reset_index(name='fn_count')
    # for each pub type, and agent type, count the total number of samples
    total_count = results.groupby(['pub_type', 'agent_type']).size().reset_index(name='total_count')
    # merge the two dataframes
    fn_count = fn_count.merge(total_count, on=['pub_type', 'agent_type'], how='left')
    # calculate the false negative rate
    # also calculate the 95% confidence interval
    fn_count['fn_rate'] = fn_count['fn_count'] / fn_count['total_count']
    fn_count['ci_lower'] = fn_count['fn_rate'] - 1.96 * np.sqrt((fn_count['fn_rate'] * (1 - fn_count['fn_rate'])) / fn_count['total_count'])
    fn_count['ci_upper'] = fn_count['fn_rate'] + 1.96 * np.sqrt((fn_count['fn_rate'] * (1 - fn_count['fn_rate'])) / fn_count['total_count'])
    fn_fp_results = pd.merge(fn_count[["pub_type", "agent_type", "fn_rate", "total_count"]], fp_count[["pub_type", "agent_type", "fp_rate"]], on=['pub_type', 'agent_type'], how='outer')
    fn_fp_results = fn_fp_results[fn_fp_results['pub_type'] != "Methods"].reset_index(drop=True)

    # Drop unnecessary index column and clean up agent_type formatting
    df = fn_fp_results
    df["agent_type"] = df["agent_type"].str.replace(r"[()]", "", regex=True)

    # Filter out "Methods" pub_type
    df = df[df["pub_type"] != "Methods"]

    # Get the sample size (n) for each pub_type
    sample_sizes = df.groupby("pub_type")["total_count"].max().to_dict()

    # Pivot the table: index=agent_type, columns=(pub_type, metric)
    pivot_df = df.pivot_table(
        index="agent_type",
        columns="pub_type",
        values=["fn_rate", "fp_rate"]
    )

    # Update the order to have fp_rate first, then fn_rate
    pivot_df = pivot_df.reindex(columns=["fp_rate", "fn_rate"], level=0)

    # Flatten the multi-index and create a new multi-header with sample sizes
    pivot_df.columns = pd.MultiIndex.from_tuples(
        [(f"{pub_type} (n={sample_sizes[pub_type]})", metric) for metric, pub_type in pivot_df.columns]
    )

    # Sort columns for readability
    pivot_df = pivot_df.sort_index(axis=1, level=0)

    # print the table
    print(pivot_df)

    # save the table to a csv file
    pivot_df.to_csv("type_i_ii_error_per_pub_type.csv")
    return fn_fp_results, results

def plot_publication_difficulty(fn_fp_results):
    # Calculate mean error rates by publication type
    df_avg = fn_fp_results.groupby('pub_type')[['fp_rate', 'fn_rate']].mean().reset_index()
    df_avg['total_error'] = df_avg['fp_rate'] + df_avg['fn_rate']
    df_avg = df_avg.sort_values('total_error', ascending=False)

    # Melt into long-form format for plotting
    df_long = fn_fp_results.melt(id_vars=['pub_type'], value_vars=['fp_rate', 'fn_rate'],
                                 var_name='Error Type', value_name='Error Rate')
    
    # Calculate confidence intervals
    ci_df = df_long.groupby(['pub_type', 'Error Type'])['Error Rate'].agg(['mean', 'count', 'std']).reset_index()
    ci_df['ci_low'] = ci_df['mean'] - 1.96 * ci_df['std'] / np.sqrt(ci_df['count'])
    ci_df['ci_high'] = ci_df['mean'] + 1.96 * ci_df['std'] / np.sqrt(ci_df['count'])

    # Ensure correct bar order
    order = df_avg['pub_type'].tolist()

    # Set improved styling for Nature with larger fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,  # Increased base font size
        'axes.labelsize': 14,  # Larger axis labels
        'axes.titlesize': 14,  # Larger title
        'xtick.labelsize': 14,  # Larger tick labels
        'ytick.labelsize': 14,
        'legend.fontsize': 14,  # Larger legend text
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3
    })

    # Create custom color palette suitable for Nature (colorblind-friendly)
    colors = ["#0173B2", "#DE8F05"]
    
    # Plot with seaborn
    plt.figure(figsize=(10, 6), dpi=300)  # Slightly larger figure size
    
    # Map the data into a format that's easier to look up
    data_map = {}
    for _, row in ci_df.iterrows():
        pub_type = row['pub_type']
        error_type = row['Error Type']
        data_map[(pub_type, error_type)] = row
        
    # Create separate plots for each error type to maintain precise control
    ax = plt.gca()
    bar_width = 0.35
    x_positions = np.arange(len(order))
    
    # First bars (fn_rate)
    fn_bars = plt.bar(x_positions - bar_width/2, 
                     [data_map.get((pub, 'fn_rate'), {}).get('mean', 0) for pub in order],
                     width=bar_width, color=colors[0], label='Type II Error')
    
    # Second bars (fp_rate)
    fp_bars = plt.bar(x_positions + bar_width/2, 
                     [data_map.get((pub, 'fp_rate'), {}).get('mean', 0) for pub in order],
                     width=bar_width, color=colors[1], label='Type I Error')
    
    # Create a list of all bars
    all_bars = fn_bars.patches + fp_bars.patches
    error_types = ['fn_rate'] * len(fn_bars) + ['fp_rate'] * len(fp_bars)
    pub_types = order * 2
    
    # Add annotations to each bar
    for i, (bar, pub_type, error_type) in enumerate(zip(all_bars, pub_types, error_types)):
        if (pub_type, error_type) in data_map:
            row = data_map[(pub_type, error_type)]
            
            # Get center of the bar
            x_pos = bar.get_x() + bar.get_width()/2
            
            # Format CI values to 2 decimal places
            mean_val = f"{row['mean']:.2f}"
            ci_low = f"{row['ci_low']:.2f}"
            ci_high = f"{row['ci_high']:.2f}"
            
            # Annotation with mean and CI interval
            value_str = f"{mean_val}\n({ci_low}-{ci_high})"
            
            # Position annotation above the bar
            y_pos = row['mean'] + 0.01
            plt.text(x_pos, y_pos, value_str,
                     ha='center', va='bottom', fontsize=11)  # Increased annotation font size

    # Set x-ticks and labels
    plt.xticks(x_positions, order, rotation=45, ha='right')
    plt.ylabel('Error Rate')
    
    # Customize legend - place in the top center of the figure
    legend = plt.legend(frameon=False, 
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.0),
                        ncol=2,  # Two columns for compact legend
                        fontsize=14,
                        fancybox=False,
                        shadow=False,
                        )  # Semi-transparent background
    
    # Adjust y-axis scale to accommodate annotations
    y_max = max([r['mean'] for r in data_map.values()]) + 0.07
    plt.ylim(0, y_max)
    
    # Add subtle grid lines on y-axis only for readability
    plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add more padding at the top for the legend
    plt.subplots_adjust(top=0.85)
    
    plt.tight_layout()
    plt.savefig("publication_difficulty.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("publication_difficulty.png", bbox_inches='tight', dpi=300)  # Also save PNG for quick viewing

def make_pairwise_comparison(fn_fp_results):
    # Set Nature-compliant styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3
    })


    # Convert to DataFrame for easier plotting
    error_df = fn_fp_results

    def get_agent_type(row):
        agent_type = row['agent_type']
        if agent_type == "reasoning_react":
            return "ReAct+Reasoning"
        elif agent_type == "react":
            return "ReAct"
        elif agent_type == "reasoning_coder":
            return "CodeGen-Reasoning"
        elif agent_type in ["coder, gpt-4o", "coder, o3-mini"]:
            return "CodeGen"
        else:
            return agent_type
        
    
    error_df['Agent Category'] = error_df.apply(get_agent_type, axis=1)
    agent_categories = error_df['Agent Category'].unique()

    error_df.rename(columns={
        'fp_rate': 'Type I Error',
        'fn_rate': 'Type II Error',
        'pub_type': 'Publication Type',
    }, inplace=True)

    # Create the overall scatter plot
    plt.figure(figsize=(8, 8))
    
    # Define markers and colors for agent categories
    markers = {
        'ReAct+Reasoning': 'o',  # Circle
        'ReAct': 's',            # Square
        'CodeGen-Reasoning': '^',        # Triangle
        'CodeGen': 'D'              # Diamond
    }
    
    colors = {
        'ReAct+Reasoning': '#0173B2',
        'ReAct': '#DE8F05',
        'CodeGen-Reasoning': '#029E73',
        'CodeGen': '#D55E00'
    }
    
    # Plot each agent category with a different marker/color
    for agent_cat in agent_categories:
        agent_data = error_df[error_df['Agent Category'] == agent_cat]
        
        # Skip if no data for this category
        if len(agent_data) == 0:
            continue
            
        # Plot points for this agent category
        plt.scatter(
            agent_data['Type I Error'], 
            agent_data['Type II Error'],
            marker=markers.get(agent_cat, 'o'),
            s=120,  # Size
            color=colors.get(agent_cat, 'blue'),
            alpha=0.8,
            label=agent_cat
        )
        
        # Add publication type annotations
        for _, row in agent_data.iterrows():
            plt.annotate(
                row['Publication Type'],
                (row['Type I Error'], row['Type II Error']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=11,
                alpha=0.8
            )
    
    # Add diagonal line for equal error rates
    max_error = max(error_df['Type I Error'].max(), error_df['Type II Error'].max())
    plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.4)
    
    # Labels and title
    plt.xlabel('Type I Error Rate')
    plt.ylabel('Type II Error Rate')
    
    # Create a legend with larger markers
    plt.legend(frameon=False, loc='lower right', markerscale=1.5)
    
    # Set equal scaling and square aspect ratio
    plt.axis('square')
    
    # Set axis limits
    plt.xlim(0.05, max_error+0.005)
    plt.ylim(0.05, max_error+0.005)
    
    # Grid lines
    plt.grid(linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save the figure
    # plt.tight_layout()
    plt.savefig('error_tradeoff_scatter.pdf', bbox_inches='tight')
    
    print("Pairwise comparison analysis complete. Files saved.")


if __name__ == "__main__":
    fn_fp_results, results = plot_type_i_ii_error_per_pub_type()
    plot_publication_difficulty(fn_fp_results)
    make_pairwise_comparison(fn_fp_results)