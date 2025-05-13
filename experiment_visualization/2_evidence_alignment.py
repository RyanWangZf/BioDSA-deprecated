import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_BASE_DIR = "/Users/zifeng/Documents/github/DSWizard"
EVIDENCE_ALIGNMENT_RESULTS_FOLDER = os.path.join(REPO_BASE_DIR, "eval_evidence_alignment")
BASE_HYPOTHESIS_PATH = os.path.join(REPO_BASE_DIR, "benchmark_datasets/cBioPortal/hypothesis")
ANALYSIS_TYPE_PATH = os.path.join(REPO_BASE_DIR, "benchmark_datasets/cBioPortal/classification")

def _get_agent_type(row) -> str:
    """
    Format agent type based on configuration.
    For react agent: (react, step_count)
    For reasoning coder: (reasoning_coder, planning_model, coding_model)
    For coder: (coder, model_name)
    """
    agent_type = row["agent_type"]
    if agent_type == "react":
        return f"(react, {str(int(row['step_count']))}, {row['model_name']})"
    elif agent_type == "reasoning_coder":
        return f"(reasoning_coder, {row['planning_model']}, {row['coding_model']})"
    elif agent_type == "coder":
        return f"(coder, {row['model_name']})"
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
    df_results['agent_type'] = df_results.apply(_get_agent_type, axis=1)
    return df_results


def _calculate_alignment_score(alignment_results):
    alignment_score = 0
    for alignment_result in alignment_results:
        if alignment_result.lower() == "supported":
            alignment_score += 1
    return alignment_score / len(alignment_results)

def _parse_completed_task(completed_task: str):
    completed_task = completed_task.strip().split("|")
    if "reasoning_coder" in completed_task or "react" in completed_task:
        agent_type = "(" + ", ".join(completed_task[-3:]) + ")"
    else:
        agent_type = "(" + ", ".join(completed_task[-2:]) + ")"
    pmid, _, hypothesis_is_true = completed_task[:3]
    return pmid, hypothesis_is_true, agent_type

def load_evidence_alignment_results():

    # load the analysis type
    df_analysis_type = pd.read_csv(os.path.join(ANALYSIS_TYPE_PATH, "analysis_classification.csv"))
    df_analysis_type["PMID"] = df_analysis_type["PMID"].astype(int).astype(str)
    df_analysis_type = df_analysis_type[["PMID", "class"]].rename(columns={"class": "analysis_type", "PMID": "pmid"})
    
    final_answer_results = get_experiment_results(os.path.join(REPO_BASE_DIR, "logs/logs_short_2"))
    final_answer_results = final_answer_results[["pmid", "hypothesis_index", "agent_type", "final_answer"]]
    with open(os.path.join(EVIDENCE_ALIGNMENT_RESULTS_FOLDER, "eval_results.json"), "r") as f:
        results = json.load(f)
    
    with open(os.path.join(EVIDENCE_ALIGNMENT_RESULTS_FOLDER, "completed_tasks.txt"), "r") as f:
        completed_tasks = f.readlines()

    output_data = []
    for result, completed_task in zip(results, completed_tasks):
        # parse completed_task
        try:
            pmid, hypothesis_is_true, agent_type = _parse_completed_task(completed_task)
        except Exception as e:
            print(f"WARNING: Error parsing completed_task: {completed_task}")
            continue

        pmid_1 = result["experiment_config"]["pmid"]
        if pmid_1 != pmid:
            raise ValueError(f"PMID mismatch: {pmid_1} != {pmid}")
        hypothesis_index = result["experiment_config"]["hypothesis_index"]
        gt_evidences = result["ground_truth_evidence"]
        generated_evidences = result["generated_evidence"]
        eval_alignment_list = result["eval_evidence_alignment"]
        alignment_results = []
        for gt_evidence_idx, gt_evidence in enumerate(gt_evidences):
            alignment_results.append(eval_alignment_list[gt_evidence_idx]['alignment'])
        alignment_score = _calculate_alignment_score(alignment_results)
        output_data.append({
            "pmid": pmid,
            "hypothesis_is_true": hypothesis_is_true,
            "hypothesis_index": hypothesis_index,
            "agent_type": agent_type,
            "alignment_results": alignment_results,
            "alignment_score": alignment_score,
        })

    df = pd.DataFrame(output_data)

    df_analysis_type = df_analysis_type.groupby("pmid")["analysis_type"].apply(lambda x: "|".join(list(set(x)))).reset_index(name="analysis_type")
    df = df.merge(df_analysis_type, on="pmid")
    return df

def plot_alignment_score_barplot(df):
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
    
    # Define a function to categorize agent types consistently
    def get_agent_type(agent_type_str):
        if 'react, 16, o3-mini' in agent_type_str:
            return 'ReAct-Reasoning'
        elif 'react' in agent_type_str:
            return 'ReAct'
        elif 'reasoning_coder' in agent_type_str:
            return 'CodeGen-Reasoning'
        elif 'coder' in agent_type_str:
            return 'CodeGen'
        return agent_type_str
    
    # Prepare the data
    df['agent_category'] = df['agent_type'].apply(get_agent_type)
    
    # Create a new column for hypothesis status that's more readable
    df['hypothesis_status'] = df['hypothesis_is_true'].apply(lambda x: 'True Hypothesis' if x=="True" else 'False Hypothesis')
    
    # Set the order of agent types
    agent_order = ['CodeGen', 'CodeGen-Reasoning', 'ReAct', 'ReAct-Reasoning']
    agent_order = [a for a in agent_order if a in df['agent_category'].unique()]
    
    # Create a color palette
    colors = {'True Hypothesis': '#0173B2', 'False Hypothesis': '#DE8F05'}
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Create a long-form DataFrame for plotting
    plot_df = df[['agent_category', 'hypothesis_status', 'alignment_score']].copy()
    
    # Calculate summary statistics for each group
    summary = plot_df.groupby(['agent_category', 'hypothesis_status']).agg(
        mean_score=('alignment_score', 'mean'),
        count=('alignment_score', 'count')
    ).reset_index()
    
    # Calculate 95% confidence intervals
    for idx, row in summary.iterrows():
        agent = row['agent_category']
        hyp = row['hypothesis_status']
        scores = plot_df[(plot_df['agent_category'] == agent) & 
                         (plot_df['hypothesis_status'] == hyp)]['alignment_score']
        
        # Standard error of the mean
        sem = scores.std() / np.sqrt(len(scores))
        # 95% confidence interval
        summary.loc[idx, 'ci_low'] = row['mean_score'] - 1.96 * sem
        summary.loc[idx, 'ci_high'] = row['mean_score'] + 1.96 * sem
    
    # Set up bar positions
    bar_width = 0.35
    x_positions = np.arange(len(agent_order))
    
    # Create bar chart
    ax = plt.gca()
    
    # Plot bars for each hypothesis status
    for i, hyp in enumerate(['True Hypothesis', 'False Hypothesis']):
        hyp_data = summary[summary['hypothesis_status'] == hyp]
        positions = [x_positions[agent_order.index(a)] + (i - 0.5) * bar_width 
                    for a in hyp_data['agent_category']]
        
        # Plot bars
        bars = ax.bar(positions, hyp_data['mean_score'], 
                     width=bar_width, color=colors[hyp], alpha=0.8,
                     label=hyp)
        
        # Add error bars
        ax.errorbar(positions, hyp_data['mean_score'],
                   yerr=[hyp_data['mean_score'] - hyp_data['ci_low'], 
                         hyp_data['ci_high'] - hyp_data['mean_score']],
                   fmt='none', color='black', capsize=3, linewidth=1)
        
        # Add value labels on top of bars
        for j, p in enumerate(positions):
            value = hyp_data.iloc[j]['mean_score']
            plt.text(p, value + 0.03, f"{value:.2f}", ha='center', fontsize=12)
    
    # Add labels and title
    plt.ylabel('Alignment Score')
    
    # Set x-axis tick positions and labels
    plt.xticks(x_positions, agent_order, rotation=30, ha='right')
    
    # Add sample size annotations
    for j, agent in enumerate(agent_order):
        for i, hyp in enumerate(['True Hypothesis', 'False Hypothesis']):
            count = len(plot_df[(plot_df['agent_category'] == agent) & 
                             (plot_df['hypothesis_status'] == hyp)])
            plt.text(x_positions[j] + (i - 0.5) * bar_width, -0.02, f"n={count}", 
                   ha='center', va='top', fontsize=12)
    
    # Add statistical significance markers
    for j, agent in enumerate(agent_order):
        true_scores = plot_df[(plot_df['agent_category'] == agent) & 
                         (plot_df['hypothesis_status'] == 'True Hypothesis')]['alignment_score']
        false_scores = plot_df[(plot_df['agent_category'] == agent) & 
                          (plot_df['hypothesis_status'] == 'False Hypothesis')]['alignment_score']
        
        if len(true_scores) > 0 and len(false_scores) > 0:
            # Perform t-test
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(true_scores, false_scores, equal_var=False)
            
            # Add significance marker
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
            else:
                sig_marker = 'ns'
            
            # Calculate max height for bracket
            # max_height = max(
            #     summary[(summary['agent_category'] == agent)]['mean_score'].max() + 
            #     summary[(summary['agent_category'] == agent)]['ci_high'].max() - 
            #     summary[(summary['agent_category'] == agent)]['mean_score'].max()
            # )
            
            # Add bracket and significance marker
            max_height = 0.25
            y_pos = max_height + 0.05
            x1 = x_positions[j] - 0.5 * bar_width
            x2 = x_positions[j] + 0.5 * bar_width
            plt.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.5)
            plt.text(x_positions[j], y_pos + 0.01, sig_marker, ha='center', va='bottom', fontsize=14)
    
    # Add grid lines on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['True Hypothesis'], alpha=0.8, label='True Hypothesis'),
                     Patch(facecolor=colors['False Hypothesis'], alpha=0.8, label='False Hypothesis')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Set y-axis limits to better fit the data while accommodating significance markers
    y_max = summary['mean_score'].max() + 0.15  # Add extra space for markers and labels
    plt.ylim(-0.05, y_max)
    
    # Set cleaner y-axis ticks
    plt.yticks(np.arange(0, y_max, 0.1))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('alignment_score_by_agent_hypothesis.pdf', bbox_inches='tight')
    
def plot_alignment_per_analysis_type(df):
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
    
    # Preprocess data - expand the analysis types that are combined with "|"
    expanded_data = []
    for _, row in df.iterrows():
        analysis_types = row['analysis_type'].split('|')
        for analysis_type in analysis_types:
            new_row = row.copy()
            new_row['analysis_type'] = analysis_type.strip()
            expanded_data.append(new_row)
    
    # Create expanded dataframe
    expanded_df = pd.DataFrame(expanded_data)
    
    # Calculate statistics by analysis type
    analysis_stats = expanded_df.groupby('analysis_type').agg(
        mean_score=('alignment_score', 'mean'),
        count=('alignment_score', 'count'),
        std=('alignment_score', 'std')
    ).reset_index()
    
    # Calculate 95% confidence intervals
    analysis_stats['ci'] = 1.96 * analysis_stats['std'] / np.sqrt(analysis_stats['count'])
    analysis_stats['ci_low'] = analysis_stats['mean_score'] - analysis_stats['ci']
    analysis_stats['ci_high'] = analysis_stats['mean_score'] + analysis_stats['ci']
    
    # Sort by mean score for better visualization
    analysis_stats = analysis_stats.sort_values(by='mean_score', ascending=False)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Set bar positions
    x_positions = np.arange(len(analysis_stats))
    
    # Create bars
    bars = plt.bar(x_positions, analysis_stats['mean_score'], 
                  width=0.6, color='#0173B2', alpha=0.8)
    
    # Add error bars
    plt.errorbar(x_positions, analysis_stats['mean_score'],
               yerr=[analysis_stats['mean_score'] - analysis_stats['ci_low'], 
                     analysis_stats['ci_high'] - analysis_stats['mean_score']],
               fmt='none', color='black', capsize=4, linewidth=1.2)
    
    # Add value annotations on top of bars
    for i, bar in enumerate(bars):
        mean_val = analysis_stats.iloc[i]['mean_score']
        ci_low = analysis_stats.iloc[i]['ci_low']
        ci_high = analysis_stats.iloc[i]['ci_high']
        plt.text(i, mean_val + 0.02, 
                f"{mean_val:.2f}\n({ci_low:.2f}-{ci_high:.2f})", 
                ha='center', va='bottom', fontsize=10)
    
    # Set axis labels
    plt.ylabel('Alignment Score')
    
    # Create x-tick labels with sample sizes
    labels = [f"{t}\n(n={c})" for t, c in zip(analysis_stats['analysis_type'], 
                                            analysis_stats['count'])]
    plt.xticks(x_positions, labels, rotation=45, ha='right')
    
    # Add grid lines on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Set y-axis limits
    plt.ylim(0, 0.35)
    
    # Add descriptive title
    # plt.title('Alignment Score by Analysis Type', fontweight='bold')
    
    # Add tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('alignment_score_by_analysis_type.pdf', bbox_inches='tight')
    plt.savefig('alignment_score_by_analysis_type.png', bbox_inches='tight', dpi=300)
    
    print(f"Analysis type plot complete. Data included {len(analysis_stats)} analysis types.")

if __name__ == "__main__":
    df = load_evidence_alignment_results()
    # plot_alignment_score_barplot(df)
    plot_alignment_per_analysis_type(df)
    pass