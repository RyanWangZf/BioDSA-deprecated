import pandas as pd
import os
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import glob
import json

def load_analysis_classification():
    output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/classification"
    df = pd.read_csv(os.path.join(output_dir, "analysis_classification.csv"))
    return df


def load_publication_classification():
    output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/classification"
    df = pd.read_csv(os.path.join(output_dir, "publication_classification.csv"))
    return df


def plot_publication_analysis_distribution(output_dir):
    # Set global plotting parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12
    })
    sns.set_style("whitegrid")

    df_analysis = load_analysis_classification()
    df_analysis["PMID"] = df_analysis["PMID"].astype(int).astype(str)
    df_analysis = df_analysis[["PMID", "class"]].rename(columns={"class": "analysis_class"})
    df_publication = load_publication_classification()
    df_publication["PMID"] = df_publication["PMID"].astype(int).astype(str)
    df_publication = df_publication[["PMID", "class"]].rename(columns={"class": "publication_class"})
    df_merged = pd.merge(df_analysis, df_publication, on="PMID")

    # group publication classes "Methods", "Pan-Cancer" together to be "Others"
    df_merged["publication_class"] = df_merged["publication_class"].replace(["Methods", "Pan-Cancer"], "Others")
    
    # Count the occurrences for each combination
    cross_tab = pd.crosstab(df_merged["publication_class"], df_merged["analysis_class"])
    
    # Calculate the total frequency for each publication type and sort
    pub_totals = cross_tab.sum(axis=1).sort_values(ascending=False)
    
    # Reorder the crosstab rows based on publication frequency
    cross_tab = cross_tab.loc[pub_totals.index]
    
    # Calculate overall frequency of each analysis type across all publications
    analysis_totals = cross_tab.sum(axis=0).sort_values(ascending=False)
    
    # Reorder the columns (analysis types) based on frequency
    cross_tab = cross_tab[analysis_totals.index]
    
    # Create a custom improved color palette
    colors = sns.color_palette("viridis", n_colors=len(cross_tab.columns))

    # Figure: Grouped bar chart for comparison
    plt.figure(figsize=(14, 10))
    
    # Plot with wider bars (width=0.9 instead of 0.8)
    cross_tab.plot(
        kind="bar", 
        color=colors,
        width=0.9
    )
    
    # Remove xlabel or set it to empty
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    
    # Move legend inside the figure - top right
    plt.legend(
        title="Analysis types", 
        loc='upper right',  # Position inside the figure
        framealpha=0.9,     # Make background slightly transparent
        edgecolor='lightgray'  # Add edge for better visibility
    )
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "analysis_distribution_grouped.pdf"), bbox_inches="tight")
    plt.close()

def plot_data_file_distribution(output_dir):
    # load the data file distribution
    # df_study = pd.read_csv("~/DSWizard/benchmark_datasets/cBioPortal/study_and_dataset_url.csv")
    
    # Set global plotting parameters for consistency with publication figures
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 24,  # Increased base font size
        'axes.labelsize': 26,  # Increased for better readability
        'axes.titlesize': 28,  # Increased for better readability
        'xtick.labelsize': 24,  # Increased for better readability
        'ytick.labelsize': 24,  # Increased for better readability
        'legend.fontsize': 20,  # Increased for better readability
        'legend.title_fontsize': 22  # Increased for better readability
    })
    sns.set_style("whitegrid")
    
    data_metadata_files = glob.glob("~/DSWizard/benchmark_datasets/cBioPortal/dataset_metadata/*.json")    
    df_data = []
    for file in data_metadata_files:
        with open(file, "r") as f:
            data = json.load(f)
            dataset_id = data["dataset_id"]
            tables = data["tables"]
            for table in tables:
                df_data.append({
                    "dataset_id": dataset_id,
                    "table_name": table["name"],
                    "n_rows": table["n_rows"],
                    "n_columns": table["n_columns"],
                })
    df_data = pd.DataFrame(df_data)
    
    # Original detailed table type classification
    def get_detailed_table_type(table_name):
        tn = table_name.lower()  # convert to lowercase for case-insensitive matching
        if "clinical" in tn:
            return "Clinical Data"
        elif "mutsig" in tn:
            return "Mutational Signficance Analysis"
        elif "mutations" in tn:
            return "Mutation Data"
        elif "gene_panel" in tn:
            return "Gene Panel"
        elif "gistic" in tn:
            return "Copy Number Alteration"
        elif "cna" in tn:
            return "Copy Number Alteration"
        elif "sv" in tn:
            return "Structural Variation"
        elif "mrna_seq" in tn or "mrna_agilent_microarray" in tn:
            return "Gene Expression"
        elif "mirna" in tn:
            return "miRNA Expression"
        elif "rppa" in tn:
            return "Protein Expression"
        elif "timeline" in tn:
            return "Patient Timeline"
        elif "methylation" in tn:
            return "Methylation"
        else:
            return "Other"
    
    # Simplified table type classification
    def get_simplified_table_type(table_name):
        detailed_type = get_detailed_table_type(table_name)
        # Merge less common types into "Other"
        if detailed_type in ["miRNA Expression", "Methylation", "Other"]:
            return "Other"
        return detailed_type
    
    # Store both detailed and simplified classifications
    df_data['detailed_type'] = df_data['table_name'].apply(get_detailed_table_type)
    df_data['table_type'] = df_data['table_name'].apply(get_simplified_table_type)
    
    # Group by table_type and calculate stats for main visualization
    table_stats = df_data.groupby('table_type').agg({
        'n_rows': 'mean',
        'n_columns': 'mean',
        'dataset_id': 'count'
    }).reset_index()
    
    table_stats.rename(columns={'dataset_id': 'frequency'}, inplace=True)
    
    # Calculate stats for types merged into "Other" for the breakout panel
    other_type_stats = df_data[df_data['detailed_type'].isin(['miRNA Expression', 'Methylation', 'Other'])].groupby('detailed_type').agg({
        'dataset_id': 'count'
    }).reset_index()
    other_type_stats.rename(columns={'dataset_id': 'frequency', 'detailed_type': 'table_type'}, inplace=True)
    other_type_stats = other_type_stats.sort_values('frequency', ascending=False)
    
    # Sort by frequency for consistent coloring
    table_stats = table_stats.sort_values('frequency', ascending=False)
    
    # Create the scatter plot with larger figure size for better layout
    fig = plt.figure(figsize=(18, 12))
    
    # Create main axis for scatter plot (with some room on the right for the breakout panel)
    main_ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)
    
    # Set up a colorblind-friendly palette with enough colors
    colors = sns.color_palette("viridis", n_colors=len(table_stats))
    
    # Calculate appropriate scaling factor based on frequency range - increase base size
    max_freq = table_stats['frequency'].max()
    min_freq = table_stats['frequency'].min()
    base_size = 5e3  # Increased base size for better visibility
    scaling_factor = 2e4 / max_freq if max_freq > 0 else 1e3  # Increased scaling for better prominence
    
    # Create scatter plot first without annotations
    for i, (idx, row) in enumerate(table_stats.iterrows()):
        # Size based on frequency with improved scaling
        size = base_size + (row['frequency'] * scaling_factor)
        
        main_ax.scatter(
            row['n_rows'], 
            row['n_columns'], 
            s=size, 
            color=colors[i], 
            alpha=0.85,  # Slightly increased alpha for better visibility 
            edgecolors='black', 
            linewidth=2.0,  # Increased linewidth for better definition
            zorder=3,
            label=row['table_type']
        )
    
    # Convert x and y to log scale after plotting
    main_ax.set_xscale('log')
    main_ax.set_yscale('log')
    
    # Function to check if two boxes overlap
    def boxes_overlap(box1, box2, buffer=15):  # Increased buffer for more spacing
        # For Rectangle objects, properly get coordinates
        box1_x0 = box1.get_x() - buffer
        box1_y0 = box1.get_y() - buffer
        box1_width = box1.get_width() + 2*buffer
        box1_height = box1.get_height() + 2*buffer
        
        box2_x0 = box2.get_x() - buffer
        box2_y0 = box2.get_y() - buffer
        box2_width = box2.get_width() + 2*buffer
        box2_height = box2.get_height() + 2*buffer
        
        # Check if one box is to the left of the other
        if box1_x0 + box1_width < box2_x0 or box2_x0 + box2_width < box1_x0:
            return False
        
        # Check if one box is above the other
        if box1_y0 + box1_height < box2_y0 or box2_y0 + box2_height < box1_y0:
            return False
        
        # If we get here, boxes overlap
        return True
    
    # Smart positioning for annotations - increased offsets for better spacing
    annotations = []
    annotation_boxes = []
    
    # Define eight possible position options around a point with increased offsets
    positions = [
        {'xytext':(40, 0), 'ha':'left', 'va':'center'},      # Right
        {'xytext':(-40, 0), 'ha':'right', 'va':'center'},    # Left
        {'xytext':(0, 40), 'ha':'center', 'va':'bottom'},    # Top
        {'xytext':(0, -40), 'ha':'center', 'va':'top'},      # Bottom
        {'xytext':(40, 40), 'ha':'left', 'va':'bottom'},     # Top-right
        {'xytext':(-40, 40), 'ha':'right', 'va':'bottom'},   # Top-left
        {'xytext':(40, -40), 'ha':'left', 'va':'top'},       # Bottom-right
        {'xytext':(-40, -40), 'ha':'right', 'va':'top'}      # Bottom-left
    ]
    
    # Add more distant positions for better separation if needed
    extended_positions = [
        {'xytext':(60, 0), 'ha':'left', 'va':'center'},      # Far Right
        {'xytext':(-60, 0), 'ha':'right', 'va':'center'},    # Far Left
        {'xytext':(0, 60), 'ha':'center', 'va':'bottom'},    # Far Top
        {'xytext':(0, -60), 'ha':'center', 'va':'top'},      # Far Bottom
        {'xytext':(60, 60), 'ha':'left', 'va':'bottom'},     # Far Top-right
        {'xytext':(-60, 60), 'ha':'right', 'va':'bottom'},   # Far Top-left
        {'xytext':(60, -60), 'ha':'left', 'va':'top'},       # Far Bottom-right
        {'xytext':(-60, -60), 'ha':'right', 'va':'top'}      # Far Bottom-left
    ]
    
    # Combine regular and extended positions
    all_positions = positions + extended_positions
    
    # Sort points by distance from center to place central annotations first
    plot_center_x = np.log10(table_stats['n_rows'].median())
    plot_center_y = np.log10(table_stats['n_columns'].median())
    
    # Calculate log coordinates for positioning
    table_stats['log_rows'] = np.log10(table_stats['n_rows'])
    table_stats['log_cols'] = np.log10(table_stats['n_columns'])
    
    # Calculate distance from center
    table_stats['dist_from_center'] = np.sqrt(
        (table_stats['log_rows'] - plot_center_x)**2 + 
        (table_stats['log_cols'] - plot_center_y)**2
    )
    
    # Sort by distance from center (closest first)
    table_stats_sorted = table_stats.sort_values('dist_from_center')
    
    # Now add annotations
    for i, (idx, row) in enumerate(table_stats_sorted.iterrows()):
        # Format for more readable labels
        label = f"{row['table_type']}\n(n={int(row['frequency'])})"
        
        # Try all positions and find the one with the least overlap
        best_position = all_positions[0]  # Default to first position
        min_overlaps = float('inf')
        
        for pos in all_positions:
            # Create temporary annotation to check positioning
            temp_ann = main_ax.annotate(
                label,
                (row['n_rows'], row['n_columns']),
                xytext=pos['xytext'],
                textcoords='offset points',
                fontsize=18,  # Increased font size for better readability
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, edgecolor='lightgray'),
                ha=pos['ha'],
                va=pos['va']
            )
            
            # Get the bounding box
            fig.canvas.draw()
            bbox = temp_ann.get_bbox_patch()
            bbox_trans = bbox.get_transform() + main_ax.transData.inverted()
            box_coords = bbox_trans.transform(bbox.get_extents())
            
            # Convert to display coordinates for easier comparison
            disp_coords = main_ax.transData.transform(box_coords)
            temp_box = plt.Rectangle((disp_coords[0, 0], disp_coords[0, 1]), 
                                     disp_coords[1, 0] - disp_coords[0, 0], 
                                     disp_coords[1, 1] - disp_coords[0, 1])
            
            # Count overlaps with existing annotations
            overlaps = sum(1 for box in annotation_boxes if boxes_overlap(temp_box, box))
            
            # If this position has fewer overlaps, use it
            if overlaps < min_overlaps:
                min_overlaps = overlaps
                best_position = pos
            
            # Remove temporary annotation
            temp_ann.remove()
            
        # Apply the best position found
        ann = main_ax.annotate(
            label,
            (row['n_rows'], row['n_columns']),
            xytext=best_position['xytext'],
            textcoords='offset points',
            fontsize=20,  # Increased font size for better readability
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, edgecolor='lightgray', linewidth=1.0),
            ha=best_position['ha'],
            va=best_position['va'],
            zorder=5
        )
        
        # Get the bounding box for the final annotation
        fig.canvas.draw()
        bbox = ann.get_bbox_patch()
        bbox_trans = bbox.get_transform() + main_ax.transData.inverted()
        box_coords = bbox_trans.transform(bbox.get_extents())
        
        # Convert to display coordinates
        disp_coords = main_ax.transData.transform(box_coords)
        box = plt.Rectangle((disp_coords[0, 0], disp_coords[0, 1]), 
                           disp_coords[1, 0] - disp_coords[0, 0], 
                           disp_coords[1, 1] - disp_coords[0, 1])
        
        annotations.append(ann)
        annotation_boxes.append(box)
    
    # Add grid lines with improved styling
    main_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add labels and title with improved styling
    main_ax.set_xlabel("Number of Rows", fontsize=24, labelpad=15)
    main_ax.set_ylabel("Number of Columns", fontsize=24, labelpad=15)
    
    # Adjust tick parameters for better readability
    main_ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    
    # Improve axis appearance
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['left'].set_linewidth(2.0)
    main_ax.spines['bottom'].set_linewidth(2.0)
    
    # Set margins
    main_ax.margins(0.15)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure with higher DPI
    plt.savefig(os.path.join(output_dir, "table_distribution.pdf"), bbox_inches="tight")
    plt.close()

def main():
    output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/eda"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # plot the distribution of publication types and analysis types
    plot_publication_analysis_distribution(output_dir)

    # plot the distribution of tables associated with each publication
    plot_data_file_distribution(output_dir)


if __name__ == "__main__":
    main()
