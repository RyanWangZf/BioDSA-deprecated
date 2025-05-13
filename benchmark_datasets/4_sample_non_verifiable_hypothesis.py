import os
import pandas as pd
import random
import json
import sys
import pdb
sys.path.append("~/DSWizard")

from src.utils.hypothesis import HypothesisLoader

# Define paths
data_dir = "~/DSWizard/benchmark_datasets/cBioPortal"
output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/non_verifiable_hypothesis"
os.makedirs(output_dir, exist_ok=True)

# Load publication and analysis data
df_publication_type = pd.read_csv(os.path.join(data_dir, "classification", "publication_classification.csv"))
df_publication_type["PMID"] = df_publication_type["PMID"].astype(int).astype(str)
# Filter out Methods papers
df_publication_type = df_publication_type[df_publication_type["class"] != "Methods"].reset_index(drop=True)
df_publication_metadata = pd.read_csv(os.path.join(data_dir, "study_metadata.csv"))
df_publication_metadata['PMID'] = df_publication_metadata['PMID'].astype(int).astype(str)
df_publication_metadata = df_publication_metadata.drop_duplicates(subset=['PMID'])
df_publication_type = df_publication_type.merge(df_publication_metadata[["PMID", "Abstract"]], on="PMID", how="left")
df_study_and_dataset = pd.read_csv(os.path.join(data_dir, "study_and_dataset_url.csv"))
df_study_and_dataset['PMID'] = df_study_and_dataset['PMID'].astype(int).astype(str)
df_study_and_dataset = df_study_and_dataset[["dataset_id","PMID"]].groupby("PMID").agg(list).reset_index()
df_study_and_dataset = df_study_and_dataset.rename(columns={"dataset_id": "dataset_ids"})
df_publication_type = df_publication_type.merge(df_study_and_dataset[["PMID", "dataset_ids"]], on="PMID", how="left")
df_publication_type = df_publication_type[df_publication_type["dataset_ids"].map(len) == 1]

# Subsample df_publication_type with the specified constraints
def stratified_subsample(df, class_col, target_size=100, min_per_class=10):
    """
    Subsample dataframe with minimum samples per class and proportional distribution
    
    Args:
        df: Input dataframe
        class_col: Column containing class labels
        target_size: Target number of samples (default: 100)
        min_per_class: Minimum samples per class (default: 10)
    
    Returns:
        Subsampled dataframe
    """
    # Get class counts and calculate proportions
    class_counts = df[class_col].value_counts()
    total_samples = len(df)
    
    # Check if we can meet the minimum per class requirement
    if min_per_class * len(class_counts) > target_size:
        raise ValueError(f"Can't satisfy constraints: {len(class_counts)} classes with min {min_per_class} each exceeds target {target_size}")
    
    # Calculate how many samples to allocate for each class
    guaranteed_samples = min_per_class * len(class_counts)
    remaining_samples = target_size - guaranteed_samples
    
    # Calculate proportional allocation of remaining samples
    proportions = class_counts / total_samples
    additional_allocation = (proportions * remaining_samples).astype(int)
    
    # Handle any rounding issues by adjusting the largest class
    if additional_allocation.sum() < remaining_samples:
        additional_allocation[additional_allocation.idxmax()] += remaining_samples - additional_allocation.sum()
    
    # Calculate final allocation for each class
    final_allocation = pd.Series(min_per_class, index=class_counts.index) + additional_allocation
    
    # Sample from each class according to the allocation
    sampled_data = []
    for class_name, count in final_allocation.items():
        class_data = df[df[class_col] == class_name]
        if len(class_data) <= count:
            # If we don't have enough samples, take all available
            sampled_data.append(class_data)
        else:
            # Sample without replacement
            sampled_data.append(class_data.sample(n=count, random_state=42))
    
    # Combine and shuffle results
    result = pd.concat(sampled_data, ignore_index=True)
    return result.sample(frac=1, random_state=42).reset_index(drop=True)

# Apply the stratified sampling
sampled_df = stratified_subsample(df_publication_type, 'class', target_size=100, min_per_class=10)
print(f"Original dataset size: {len(df_publication_type)}")
print(f"Sampled dataset size: {len(sampled_df)}")
print("Class distribution in original dataset:")
print(df_publication_type['class'].value_counts())
print("\nClass distribution in sampled dataset:")
print(sampled_df['class'].value_counts())

# Continue with the sampled dataframe for the rest of the processing
df_publication_type = sampled_df
# Load hypothesis data
hypothesis_loader = HypothesisLoader(os.path.join(data_dir, "hypothesis"))

# Group studies by publication type
publication_types = df_publication_type["class"].unique()
studies_by_type = {ptype: df_publication_type[df_publication_type["class"] == ptype]["PMID"].tolist() 
                   for ptype in publication_types}

# Process each study
processed_count = 0
for index, row in df_publication_type.iterrows():
    study_pmid = row["PMID"]
    study_type = row["class"]
    
    # Get all other studies of the same type
    other_studies = [pmid for pmid in studies_by_type[study_type] if pmid != study_pmid]
    
    if not other_studies:
        print(f"No other studies found for type {study_type}, skipping study {study_pmid}")
        continue
    
    # Randomly select another study of the same type
    other_study_pmid = random.choice(other_studies)
    
    try:
        # Load hypotheses from the other study
        other_study_data, _ = hypothesis_loader.get_hypothesis(other_study_pmid)
        
        if not other_study_data or not other_study_data.hypotheses:
            print(f"No hypotheses found for study {other_study_pmid}, skipping")
            continue
        
        # Randomly select one hypothesis from the other study
        selected_hypothesis = random.choice(other_study_data.hypotheses)
        
        # Create output data structure
        output_data = {
            "PMID": str(study_pmid),
            "Title": row["Title"],
            "Abstract": row["Abstract"],
            "dataset_ids": row["dataset_ids"],
            "hypotheses": [{
                "hypothesis": selected_hypothesis.hypothesis
            }]
        }
        
        # Save to output file
        output_file = os.path.join(output_dir, f"{study_pmid}.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} studies")
    
    except Exception as e:
        print(f"Error processing study {study_pmid} with non-verifiable study {other_study_pmid}: {str(e)}")

print(f"Completed processing. Generated {processed_count} non-verifiable hypothesis files.") 