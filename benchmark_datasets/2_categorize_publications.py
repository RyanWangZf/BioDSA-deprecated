"""
This script categorizes publications into different categories based on the content of the publications.
"""

import os
import pandas as pd
import pdb
import json
import re

# add the dswizard directory to the python path
import sys
sys.path.append("~/DSWizard")

# load the env about the openai api key
from dotenv import load_dotenv
load_dotenv("~/DSWizard/.env")

from src.llm.llm import (
    call_llm_json_output, 
    batch_call_llm_json_output
)

PROMPT = """
You are an expert in biomedical literature classification with deep domain knowledge. Your task is to classify a given publication into one of the following eight categories: "Genomics", "Molecular", "Pan-Cancer", "Therapeutics", "Biomarkers", "Methods", "Integrative", or "Translational".

For clarity, please note the following detailed descriptions for each category:

- **Genomics:** Publications that focus primarily on large-scale genomic profiling. These studies use high-throughput sequencing data to catalog genomic alterations such as mutations, copy-number variations, and structural variants. Keywords include "genome sequencing", "TCGA", "somatic mutations", etc.

- **Molecular:** Research that emphasizes molecular-level analyses beyond DNA, including transcriptomic, proteomic, and epigenomic characterization. These publications delve into gene expression, protein interactions, or regulatory mechanisms. Look for terms like "RNA sequencing", "proteomics", "epigenetics", etc.

- **Pan-Cancer:** Studies that analyze multiple cancer types to compare and contrast their molecular or genomic features. These papers highlight cross-cancer similarities or differences using integrated datasets across various tumor types. Keywords include "pan-cancer", "cross-cancer", "comparative analysis", etc.

- **Therapeutics:** Publications linking genomic or molecular data with therapeutic responses. This category covers studies investigating drug sensitivity/resistance, identifying therapeutic targets, or evaluating personalized treatment strategies. Important terms include "drug response", "targeted therapy", "clinical trials", etc.

- **Biomarkers:** Research focused on identifying diagnostic, prognostic, or predictive markers. These publications report on genetic or molecular signatures that correlate with disease outcome, progression, or treatment response. Look for "biomarker", "prognostic indicator", "diagnostic marker", etc.

- **Methods:** Studies that introduce or validate new computational, statistical, or experimental methods for analyzing biomedical data. These papers propose innovative algorithms, tools, or approaches to improve data analysis in cancer research. Keywords include "algorithm", "method development", "computational tool", etc.

- **Integrative:** Publications that combine multiple layers of data—such as genomic, transcriptomic, proteomic, and/or epigenomic—to provide a comprehensive view of cancer biology. The focus is on the synthesis of diverse data sources for deeper insights. Terms like "multi-omics", "data integration", and "comprehensive analysis" are key.

- **Translational:** Research that bridges laboratory discoveries with clinical applications. These studies focus on implementing genomic or molecular insights in clinical practice, aiming to improve diagnosis, prognosis, or treatment protocols. Keywords include "clinical translation", "precision medicine", "patient stratification", etc.

Now, given the input publication text below, carefully analyze its content. Consider all aspects such as whether it highlights genomic sequencing data, molecular experiments, cross-cancer comparisons, therapeutic implications, biomarker discovery, methodological innovations, integrative analyses, or clinical applications. Based on this analysis, assign the publication to the single category that best represents its primary focus.

# Input publication
{paper_content}

# Output
Output only a JSON dictionary in the exact following format:
```json
{{"class": <one of the eight categories>}}
```
"""


def _extract_from_code_block(text: str) -> str:
    """
    Extract content from markdown code blocks.
    
    Looks for patterns like:
    ```json
    {"key": "value"}
    ```
    
    or just:
    ```
    {"key": "value"}
    ```
    
    Args:
        text (str): Text that might contain markdown code blocks
        
    Returns:
        str: Extracted content from within code blocks, or the original text if no blocks found
    """
    # Regular expressions for different code block patterns
    patterns = [
        r'```(?:json)?\n([\s\S]*?)\n```',  # Standard markdown code blocks with optional language
        r'```(?:json)?([\s\S]*?)```',      # Code blocks without newlines
        r'`([\s\S]*?)`',                   # Inline code blocks
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the first substantial match
            for match in matches:
                if match and match.strip():
                    return match.strip()
    
    # No code blocks found, return the original text
    return text

def process_llm_output(raw_output):
    if not raw_output or not raw_output.strip():
        return None
        
    # Try multiple parsing strategies
    parsing_strategies = [
        lambda text: text,  # Original text as is
        _extract_from_code_block,  # Extract from markdown code blocks
        lambda text: text.strip().strip('"\''),  # Remove quotes and whitespace
    ]
    
    for strategy in parsing_strategies:
        try:
            text_to_parse = strategy(raw_output)
            if text_to_parse:
                parsed = json.loads(text_to_parse)
                return parsed
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Unexpected error in JSON parsing: {e}")
            continue
            
    # If we get here, all parsing strategies failed
    print(f"Error parsing JSON, all strategies failed: {raw_output[:100]}...")
    return None

def classify_publications(debug=False):
    df = pd.read_csv("~/DSWizard/benchmark_datasets/cBioPortal/study_metadata.csv")
    output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/classification"
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        df = df.iloc[0:10]

    df_study_to_dataset = df[["PMID","dataset_id"]].groupby("PMID").agg(list).reset_index()
    df_study = df[["PMID","Title","Abstract","Results"]].drop_duplicates(subset=["PMID"])
    df_study.fillna("", inplace=True)
    texts = df_study[["Title", "Abstract"]].apply(lambda x: "\n".join(x.dropna()), axis=1)
    texts = texts.to_list()
    batch_inputs = [{"paper_content": text} for text in texts]
    outputs = batch_call_llm_json_output(
        prompt_template=PROMPT,
        batch_inputs=batch_inputs,
        llm="gpt-4o-mini",
        temperature=0.0,
        batch_size=10,
        max_completion_tokens=128,
    )
    all_parsed = []
    for output in outputs:
        parsed = process_llm_output(output)
        all_parsed.append(parsed.get("class", ""))
    all_parsed = pd.DataFrame(all_parsed, columns=["class"])
    all_parsed = pd.concat([df_study[["PMID","Title"]], all_parsed], axis=1)
    all_parsed.to_csv(os.path.join(output_dir, "publication_classification.csv"), index=False)
    print(f"Saved to {os.path.join(output_dir, 'publication_classification.csv')}")

if __name__ == "__main__":
    debug = False
    classify_publications(debug=debug)