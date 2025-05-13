import pandas as pd
from typing import Dict, List, Tuple
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

ERROR_TYPE_MAPPING = {
    # Variable / Object Misuse
    "KeyError": "Variable/Object Misuse",
    "AttributeError": "Variable/Object Misuse",
    "NameError": "Variable/Object Misuse",
    "UnboundLocalError": "Variable/Object Misuse",
    "IndexError": "Variable/Object Misuse",
    "TypeError": "Variable/Object Misuse",
    "NotImplementedError": "Variable/Object Misuse",

    # Mathematical / Logical Errors
    "ZeroDivisionError": "Math/Logic Error",
    "ValueError": "Math/Logic Error",
    "AssertionError": "Math/Logic Error",
    "numpy.linalg.LinAlgError": "Math/Logic Error",
    "numpy.exceptions.AxisError": "Math/Logic Error",
    "numpy.core._exceptions._UFuncNoLoopError": "Math/Logic Error",
    "numpy.core._exceptions._ArrayMemoryError": "Math/Logic Error",
    "lifelines.exceptions.ConvergenceError": "Math/Logic Error",

    # Module / Import Issues
    "ImportError": "Import/Module Error",
    "ModuleNotFoundError": "Import/Module Error",

    # File and I/O Errors
    "FileNotFoundError": "File/I-O Error",
    "OSError": "File/I-O Error",
    "_csv.Error": "File/I-O Error",

    # Pandas / DataFrame-Specific Issues
    "pandas.errors.ParserError": "Pandas/Data Error",
    "pandas.errors.MergeError": "Pandas/Data Error",
    "pandas.errors.IndexingError": "Pandas/Data Error",
    "pandas.errors.IntCastingNaNError": "Pandas/Data Error",

    # General / Catch-all
    "Exception": "General Exception",
    "RuntimeError": "General Exception"
}


def _get_agent_type(row) -> str:
    """
    Format agent type based on configuration.
    For react agent: (react, step_count)
    For reasoning coder: (reasoning_coder, planning_model, coding_model)
    For coder: (coder, model_name)
    """
    agent_type = row["agent_type"]
    if agent_type == "react":
        return f"react, {str(int(row['step_count']))}, {row['model_name']}"
    elif agent_type == "reasoning_coder":
        return f"reasoning_coder, {row['planning_model']}, {row['coding_model']}"
    elif agent_type == "coder":
        return f"coder, {row['model_name']}"
    return agent_type

def get_standard_agent_type(x):
    if x.startswith('coder'):
        return 'CodeGen'
    elif x.startswith('reasoning_coder'):
        return 'CodeGen-Reasoning'
    elif "react, 16, gpt-4o" in x:
        return 'ReAct'
    elif "react, 16, o3-mini" in x:
        return 'ReAct-Reasoning'
    return x

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

def extract_error_metadata(observations: str) -> dict:
    """
    Extract error metadata from observations containing stack traces.
    
    Args:
        observations: String containing the execution output
        
    Returns:
        Dictionary with error metadata including:
        - has_error: Boolean indicating if a traceback was found
        - error_type: The type of exception if found (e.g., KeyError, ValueError)
        - error_message: The error message if found
    """
    import re
    
    result = {
        "has_error": False,
        "error_type": None,
        "error_message": None
    }
    
    # Check if there's a traceback in the observations
    if 'Traceback' in observations:
        result["has_error"] = True
        
        # Try to extract the error type and message using regex
        # This pattern looks for the error type at the end of a traceback
        # Updated to handle module-specific errors like _csv.Error
        error_pattern = r'Traceback \(most recent call last\):.*?([A-Za-z0-9_\.]+(?:Error|Exception)):\s*([^\n]+)'
        match = re.search(error_pattern, observations, re.DOTALL)
        
        if match:
            result["error_type"] = match.group(1)
            result["error_message"] = match.group(2).strip()
        else:
            # If we found a traceback but couldn't parse the error type, log a warning
            print(f"WARNING: Traceback found but couldn't parse error type from: {observations[:200]}...")
            
            # Try a more lenient pattern to catch other exception types
            alt_pattern = r'Traceback \(most recent call last\):.*?([A-Za-z0-9_\.]+):\s*([^\n]+)'
            alt_match = re.search(alt_pattern, observations, re.DOTALL)
            
            if alt_match:
                result["error_type"] = alt_match.group(1)
                result["error_message"] = alt_match.group(2).strip()
    
    return result

def extract_react_code_cells(observations: str) -> List[Tuple[int, str]]:
    """
    Extract code cells and their corresponding stdout from React agent observations.
    
    Args:
        observations: String containing the React agent observations
        
    Returns:
        List of tuples (observation_number, stdout) for each code cell
    """
    import re
    
    # Pattern to match observation blocks
    observation_pattern = r'## Observation (\d+)\n### Code: \n```.*?\n(.*?)\n```\n### Stdout:\n(.*?)(?=\n## Observation|\Z)'
    
    # Find all matches
    matches = re.findall(observation_pattern, observations, re.DOTALL)
    
    # Extract observation number and stdout for each match
    result = []
    for match in matches:
        observation_num = int(match[0])
        code = match[1]
        stdout = match[2]
        result.append((observation_num, stdout))
    
    return result

def process_observations(row_dict: Dict) -> Dict:
    """
    Process the observations column to extract the number of code cells and the number of executable code cells
    and extract error metadata if present.
    """
    agent_type = row_dict['agent_type']
    observations = row_dict['observations']
    total_code_cells = 0
    executable_code_cells = 0
    
    if 'coder' in agent_type or 'reasoning_coder' in agent_type:
        total_code_cells = 1
        
        # Extract error metadata
        error_data = extract_error_metadata(observations)
        
        # Add error metadata to the row dictionary
        row_dict['has_error'] = error_data['has_error']
        row_dict['error_type'] = ERROR_TYPE_MAPPING.get(error_data['error_type'], "Unknown Error")
        row_dict['error_message'] = error_data['error_message']
        
        if not error_data['has_error']:
            executable_code_cells = 1
        
    elif 'react' in agent_type:
        # Extract code cells and their stdout
        code_cells = extract_react_code_cells(observations)
        
        # Calculate total code cells
        total_code_cells = len(code_cells)
        
        # Check each stdout for errors
        error_types = []
        error_messages = []
        
        for _, stdout in code_cells:
            error_data = extract_error_metadata(stdout)
            if not error_data['has_error']:
                executable_code_cells += 1
            else:
                if error_data['error_type']:
                    error_types.append(ERROR_TYPE_MAPPING.get(error_data['error_type'], "Unknown Error"))
                if error_data['error_message']:
                    error_messages.append(error_data['error_message'])
        
        # Store error information
        row_dict['has_error'] = total_code_cells > executable_code_cells
        row_dict['error_type'] = error_types
        row_dict['error_message'] = error_messages
    
    row_dict['total_code_cells'] = total_code_cells
    row_dict['executable_code_cells'] = executable_code_cells
    
    return row_dict

# Function to analyze executability rate by agent type
def analyze_executability_rate(df):
    """
    Analyze the executability rate by agent type.
    
    Args:
        df: DataFrame containing the processed observations
        
    Returns:
        DataFrame with executability rate by agent type
    """
    # Process all observations
    processed_df = df.apply(process_observations, axis=1)

    # check that the index is not duplicated
    assert not processed_df.index.duplicated().any(), "Index is duplicated"
    
    # Group by agent type and calculate executability rate
    executability = processed_df.groupby('agent_type').apply(
        lambda x: {
            'total_code_cells': x['total_code_cells'].sum(),
            'executable_code_cells': x['executable_code_cells'].sum(),
            'executability_rate': x['executable_code_cells'].sum() / x['total_code_cells'].sum() if x['total_code_cells'].sum() > 0 else 0
        }, include_groups=False
    ).apply(pd.Series)
    
    return executability

# Analyze error types across all agent types
def analyze_error_types(df):
    """
    Analyze the types of errors across the dataset.
    
    Args:
        df: DataFrame containing the processed observations
        
    Returns:
        DataFrame with error type counts
    """
    # Process all observations to extract error metadata
    processed_df = df.apply(process_observations, axis=1)
    # processed_df["error_type"] = processed_df["error_type"].apply(lambda x: ERROR_TYPE_MAPPING.get(x, "Unknown Error"))
    
    # For coder and reasoning_coder agents
    standard_agents = processed_df[
        (processed_df['agent_type'].str.contains('coder|reasoning_coder')) & 
        (processed_df['has_error'] == True)
    ]
    
    error_counts = standard_agents['error_type'].value_counts().reset_index()
    error_counts.columns = ['Error Type', 'Count']
    
    # For react agents, we need to flatten the list of error types
    react_agents = processed_df[processed_df['agent_type'].str.contains('react')]
    react_error_types = []
    
    for error_types in react_agents['error_type'].dropna():
        react_error_types.extend(error_types)
    
    if react_error_types:
        react_error_counts = pd.Series(react_error_types).value_counts().reset_index()
        react_error_counts.columns = ['Error Type', 'Count']
        
        # Combine the error counts
        error_counts = pd.concat([error_counts, react_error_counts]).groupby('Error Type').sum().reset_index()
    
    return error_counts

def analyze_error_types_single_record(df):
    """
    Analyze the types of errors across the dataset.
    
    Args:
        df: DataFrame containing the processed observations
        
    Returns:
        DataFrame with error type counts
    """
    # Process all observations to extract error metadata
    processed_df = df.apply(process_observations, axis=1)

    processed_df = processed_df[["agent_type", "pmid", "hypothesis_index","hypothesis","hypothesis_is_true", "final_answer","has_error", "total_code_cells", "executable_code_cells"]]

    processed_df["agent_type"] = processed_df["agent_type"].apply(lambda x:
    
    {
        "react, 16, gpt-4o": "ReAct",
        "react, 16, o3-mini": "ReAct-Reasoning",
        "coder, gpt-4o": "CodeGen-gpt-4o",
        "coder, o3-mini": "CodeGen-o3-mini",
        "reasoning_coder, o3-mini, gpt-4o": "CodeGen-Reasoning",
    }[x]
    )

    return processed_df


def get_error_counts_and_executability_analysis(df_results):
    """
    Categorize error types based on the error message.
    """
    executability_analysis = analyze_executability_rate(df_results)
    executability_analysis.reset_index(inplace=True)
    executability_analysis["agent_type"] = executability_analysis["agent_type"].apply(get_standard_agent_type)
    overall_error_counts = []
    agent_type_list = df_results['agent_type'].unique()
    for agent_type in agent_type_list:
        df_results_agent = df_results[df_results['agent_type'] == agent_type]
        error_counts = analyze_error_types(df_results_agent)
        error_counts["agent_type"] = agent_type
        overall_error_counts.append(error_counts)
    overall_error_counts = pd.concat(overall_error_counts)
    overall_error_counts["agent_type"] = overall_error_counts["agent_type"].apply(get_standard_agent_type)
    return overall_error_counts, executability_analysis


def plot_executability_and_errors(df_results):
    """
    Create a combined visualization showing executability rates and error type distributions by agent type.
    """
    # Set Nature-compliant styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 14,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4
    })
    
    # Get executability analysis and error counts
    overall_error_counts, executability_analysis = get_error_counts_and_executability_analysis(df_results)
    
    # Deal with duplicate entries by aggregating
    # First, ensure each agent type appears once in executability_analysis
    executability_analysis = executability_analysis.groupby('agent_type').agg({
        'total_code_cells': 'sum',
        'executable_code_cells': 'sum',
        'executability_rate': 'mean'  # We'll recalculate this
    }).reset_index()
    
    # Recalculate executability rate after aggregation
    executability_analysis['executability_rate'] = (
        executability_analysis['executable_code_cells'] / executability_analysis['total_code_cells']
    )
    
    # Aggregate error counts by agent_type and Error Type
    overall_error_counts = overall_error_counts.groupby(['agent_type', 'Error Type'])['Count'].sum().reset_index()
    
    # Define the order of agent types (from left to right)
    agent_order = ['CodeGen', 'CodeGen-Reasoning', 'ReAct', 'ReAct-Reasoning']
    
    # Filter to include only agent types in our order
    executability_analysis = executability_analysis[executability_analysis['agent_type'].isin(agent_order)]
    
    # Sort by predefined order
    executability_analysis['order'] = executability_analysis['agent_type'].apply(lambda x: agent_order.index(x))
    executability_analysis = executability_analysis.sort_values('order')
    executability_analysis = executability_analysis.drop('order', axis=1)
    
    # Create a pivot table for error types
    error_pivot = pd.pivot_table(
        overall_error_counts,
        index='agent_type', 
        columns='Error Type', 
        values='Count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Ensure error_pivot has all agent types in our order
    for agent in agent_order:
        if agent not in error_pivot.index:
            error_pivot.loc[agent] = 0
    
    # Sort by predefined order
    error_pivot = error_pivot.loc[agent_order]
    
    # Calculate percentages for error types
    error_percentages = error_pivot.copy()
    for idx in error_percentages.index:
        total_cells = executability_analysis.loc[executability_analysis['agent_type'] == idx, 'total_code_cells'].values[0]
        non_executable = total_cells - executability_analysis.loc[executability_analysis['agent_type'] == idx, 'executable_code_cells'].values[0]
        
        if non_executable > 0:
            # Calculate percentages relative to total cells
            error_percentages.loc[idx] = error_percentages.loc[idx] / total_cells * 100
            
            # Ensure error percentages sum to the non-executable percentage
            error_sum = error_percentages.loc[idx].sum()
            if error_sum > 0:
                scaling_factor = (100 - executability_analysis.loc[executability_analysis['agent_type'] == idx, 'executability_rate'].values[0] * 100) / error_sum
                error_percentages.loc[idx] = error_percentages.loc[idx] * scaling_factor
    
    # Group very small error types into "Other Errors" to enhance visibility
    VISIBILITY_THRESHOLD = 1.5  # Minimum percentage to display as separate category
    
    # Identify minor error types
    minor_error_types = []
    for col in error_percentages.columns:
        if error_percentages[col].max() < VISIBILITY_THRESHOLD:
            minor_error_types.append(col)
    
    # If we have minor error types, group them
    if minor_error_types:
        # Create "Other Errors" column
        error_percentages['Other Errors'] = error_percentages[minor_error_types].sum(axis=1)
        error_pivot['Other Errors'] = error_pivot[minor_error_types].sum(axis=1)
        
        # Drop the original minor columns
        error_percentages = error_percentages.drop(columns=minor_error_types)
        error_pivot = error_pivot.drop(columns=minor_error_types)
    
    # Create figure with better proportions
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Define an improved color palette that's more distinguishable
    executable_color = '#2c7fb8'  # Strong blue for executable
    
    # Define a color palette with better distinction
    error_palette = {
        'Variable/Object Misuse': '#e34a33',   # Vibrant red
        'Math/Logic Error': '#41ab5d',         # Strong green
        'Import/Module Error': '#fdae61',      # Warm amber
        'Pandas/Data Error': '#7b3294',        # Purple
        'File/I-O Error': '#c51b7d',           # Magenta
        'General Exception': '#636363',        # Dark gray
        'Other Errors': '#969696'              # Medium gray
    }
    
    # Ensure any error types not explicitly defined get assigned colors
    for error_type in error_percentages.columns:
        if error_type not in error_palette and error_type != 'Other Errors':
            error_palette[error_type] = '#8c96c6'  # Default to a pale purple
    
    # Plot with better spacing between bars
    x = np.arange(len(agent_order))
    width = 0.7  # Slightly narrower bars for better spacing
    
    # Plot executable portion first (bottom of stack)
    bars = plt.bar(x, executability_analysis['executability_rate'] * 100, 
                  color=executable_color, alpha=1.0, width=width,
                  label='Executable', edgecolor='white', linewidth=0.6)
    
    # Add value labels inside the executable bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 15:  # Only add text if bar is tall enough
            plt.text(i, height/2, f'{height:.1f}%', 
                    ha='center', va='center', color='white', 
                    fontweight='bold', fontsize=18)
    
    # Plot error types on top of executable portion
    bottom = executability_analysis['executability_rate'] * 100
    
    # Sort error types by average percentage (descending) for better visualization
    error_importance = error_percentages.mean().sort_values(ascending=False).index
    
    for error_type in error_importance:
        values = error_percentages[error_type].values
        
        # Skip error types with very low values across all agent types
        if values.max() < 0.5:
            continue
            
        # Get color for this error type
        color = error_palette.get(error_type, '#8c96c6')
        
        bars = plt.bar(x, values, bottom=bottom, 
                      color=color, 
                      alpha=1.0, width=width, label=error_type,
                      edgecolor='white', linewidth=0.6)
        
        # Add value labels inside the bars for segments that are large enough
        for i, v in enumerate(values):
            if v >= 6:  # Only label if segment is at least 6%
                plt.text(x[i], bottom[i] + v/2, f'{v:.1f}%', 
                        ha='center', va='center', color='white', 
                        fontweight='bold', fontsize=18)
        
        # Update bottom for next stack
        bottom += values
    
    # Add sample sizes with better positioning
    # for i, agent in enumerate(agent_order):
    #     cell_count = executability_analysis.loc[executability_analysis['agent_type'] == agent, 'total_code_cells'].values[0]
    #     plt.text(i, -15, f'n={int(cell_count)}', ha='center', fontsize=14)
    
    # Set labels and styling
    plt.ylabel('Percentage (%)', fontweight='bold', fontsize=20)
    
    # Customize x-axis labels with better formatting
    agent_labels = {
        'CodeGen': 'CodeGen',
        'CodeGen-Reasoning': 'CodeGen\nReasoning',
        'ReAct': 'ReAct',
        'ReAct-Reasoning': 'ReAct\nReasoning'
    }
    plt.xticks(x, [agent_labels[agent] for agent in agent_order], rotation=0, fontsize=20)
    plt.ylim(0, 105)  # Add room for annotations
    
    # Add more padding at the bottom for sample sizes
    plt.subplots_adjust(bottom=0.20)
    
    # Add subtle grid lines on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.2, color='gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add thin border around the plot for Nature style
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    # Create legend with better positioning and formatting
    # Reorder legend items to put 'Executable' first
    handles, labels = ax.get_legend_handles_labels()
    
    if 'Executable' in labels:
        exec_idx = labels.index('Executable')
        handles = [handles[exec_idx]] + [h for i, h in enumerate(handles) if i != exec_idx]
        labels = [labels[exec_idx]] + [l for i, l in enumerate(labels) if i != exec_idx]
    
    # Place legend at bottom with two rows for better layout
    legend = plt.legend(handles, labels, 
                       loc='upper center', 
                       bbox_to_anchor=(0.5, -0.15), 
                       ncol=3, frameon=False, fontsize=20,
                       framealpha=0.9, edgecolor='lightgray')
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig('executability_and_errors.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('executability_and_errors.png', bbox_inches='tight', dpi=300)
    
    print("Executability and error type visualization complete with optimized styling.")


def plot_hypothesis_decision_for_non_executable_code(df_results):
    """
    Plot the hypothesis decision distribution comparing cases with and without execution errors across agent types.
    Shows how execution failures impact the final answer distribution.
    """
    # Process and prepare the data
    df_results = analyze_error_types_single_record(df_results)
    df_results = df_results[df_results['agent_type'].apply(lambda x: "CodeGen" in x)].reset_index(drop=True)
    
    # Standardize final_answer values
    df_results['final_answer'] = df_results["final_answer"].apply(lambda x: {
        "True - the hypothesis is supported by the data": "True",
    }.get(x, x))
    
    # Create a figure with Nature-compliant styling
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14
    })
    
    # Define agent types and colors
    agent_types = ['CodeGen-gpt-4o', 'CodeGen-o3-mini', 'CodeGen-Reasoning']
    error_colors = {'True': '#fc8d59', 'False': '#91bfdb', 'Not Verifiable': '#969696'}
    
    # Set up the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Plot for data with execution errors
    ax1 = axs[0]
    plot_data_with_errors = df_results[df_results['has_error'] == True]
    
    # Create a pivot table for data with errors
    error_pivot = pd.pivot_table(
        plot_data_with_errors, 
        index='agent_type',
        columns='final_answer', 
        values='pmid', 
        aggfunc='count',
        fill_value=0
    )
    
    # Ensure all agent types are present
    for agent in agent_types:
        if agent not in error_pivot.index:
            error_pivot.loc[agent] = 0
    
    # Ensure all final answer types are present
    for answer in ['True', 'False', 'Not Verifiable']:
        if answer not in error_pivot.columns:
            error_pivot[answer] = 0
    
    # Calculate percentages
    error_percentages = error_pivot.div(error_pivot.sum(axis=1), axis=0) * 100
    
    # Plot stacked bars for error cases
    bottom = np.zeros(len(agent_types))
    for answer in ['True', 'False', 'Not Verifiable']:
        if answer in error_percentages.columns:
            values = [error_percentages.loc[agent, answer] if agent in error_percentages.index else 0 for agent in agent_types]
            ax1.bar(agent_types, values, bottom=bottom, label=answer if bottom.sum() == 0 else "", color=error_colors[answer])
            
            # Add percentage labels
            for i, v in enumerate(values):
                if v >= 5:  # Only show percentages for significant segments
                    ax1.text(i, bottom[i] + v/2, f'{v:.1f}%', ha='center', va='center', color='black', fontweight='bold')
            
            bottom += values
    
    # Annotate the total counts
    for i, agent in enumerate(agent_types):
        if agent in error_pivot.index:
            count = error_pivot.loc[agent].sum()
            ax1.text(i, 105, f'n={int(count)}', ha='center', fontsize=18)
    
    ax1.set_title('Non-Executable Code', fontsize=22)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Percentage (%)', fontsize=18)
    ax1.set_xticks(range(len(agent_types)))
    ax1.set_xticklabels([t.replace('CodeGen-', 'CodeGen\n') for t in agent_types], rotation=0)
    
    # Plot for data without execution errors
    ax2 = axs[1]
    plot_data_no_errors = df_results[df_results['has_error'] == False]
    
    # Create a pivot table for data without errors
    no_error_pivot = pd.pivot_table(
        plot_data_no_errors, 
        index='agent_type',
        columns='final_answer', 
        values='pmid', 
        aggfunc='count',
        fill_value=0
    )
    
    # Ensure all agent types are present
    for agent in agent_types:
        if agent not in no_error_pivot.index:
            no_error_pivot.loc[agent] = 0
    
    # Ensure all final answer types are present
    for answer in ['True', 'False', 'Not Verifiable']:
        if answer not in no_error_pivot.columns:
            no_error_pivot[answer] = 0
    
    # Calculate percentages
    no_error_percentages = no_error_pivot.div(no_error_pivot.sum(axis=1), axis=0) * 100
    
    # Plot stacked bars for no error cases
    bottom = np.zeros(len(agent_types))
    for answer in ['True', 'False', 'Not Verifiable']:
        if answer in no_error_percentages.columns:
            values = [no_error_percentages.loc[agent, answer] if agent in no_error_percentages.index else 0 for agent in agent_types]
            ax2.bar(agent_types, values, bottom=bottom, label=answer, color=error_colors[answer])
            
            # Add percentage labels
            for i, v in enumerate(values):
                if v >= 5:  # Only show percentages for significant segments
                    ax2.text(i, bottom[i] + v/2, f'{v:.1f}%', ha='center', va='center', color='black', fontweight='bold')
            
            bottom += values
    
    # Annotate the total counts
    for i, agent in enumerate(agent_types):
        if agent in no_error_pivot.index:
            count = no_error_pivot.loc[agent].sum()
            ax2.text(i, 105, f'n={int(count)}', ha='center', fontsize=18)
    
    ax2.set_title('Executable Code', fontsize=22)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(range(len(agent_types)))
    ax2.set_xticklabels([t.replace('CodeGen-', 'CodeGen\n') for t in agent_types], rotation=0)
    
    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, frameon=False, fontsize=20)
    
    # Add overall title
    # plt.suptitle('Hypothesis Decision Distribution by Error Status', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)
    
    # Save figures
    plt.savefig('hypothesis_decision_by_error.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('hypothesis_decision_by_error.png', bbox_inches='tight', dpi=300)
    
    print("Hypothesis decision visualization by error status completed.")


if __name__ == "__main__":
    df_results = get_experiment_results(os.path.join(REPO_BASE_DIR, "logs/logs_short_2"))

    # Run the analysis
    # plot_executability_and_errors(df_results)
    plot_hypothesis_decision_for_non_executable_code(df_results)
    
    print("Analysis completed successfully.")