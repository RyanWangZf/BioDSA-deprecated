"""
This script categorizes publications into different categories based on the content of the publications.
"""

import os
import pandas as pd
import pdb
import json
import re
from glob import glob

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

PROMPT = """You are an expert in biomedical computational literature. Your task is to classify a given analysis task into one of the following eight categories: "Correlation", "Comparison", "Frequency", "Clustering", "Survival", "Functional", "Structural", or "Pathway".

For clarity, here are detailed explanations for each category:
- Correlation: Tasks that focus on establishing statistical associations between variables.
- Comparison: Tasks that directly compare groups or conditions.
- Frequency: Tasks that quantify the occurrence or prevalence of specific events.
- Clustering: Tasks that group data based on similarity or patterns.
- Survival: Tasks that analyze and relate data to patient outcomes.
- Functional: Tasks that assess gene function or cellular behavior experimentally.
- Structural: Tasks that explore genomic or structural changes in the data.
- Pathway: Tasks that integrate multiple types of data to elucidate biological pathways.

Now, analyze the input analysis text carefully and decide which category best describes the task.

# Input analysis
{analysis_text}

# Output
Output only a JSON dictionary in the exact following format:
```json
{{"class": "<one of the eight categories>"}}
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


def categorize_analysis(debug=False):
    output_dir = "~/DSWizard/benchmark_datasets/cBioPortal/classification"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hypothesis_files = glob("~/DSWizard/benchmark_datasets/cBioPortal/hypothesis/*.json")
    analyses = []
    for hypothesis_file in hypothesis_files:
        with open(hypothesis_file, "r") as f:
            data = json.load(f)
        hypotheses = data["hypotheses"]
        for hypothesis in hypotheses:
            for evidence in hypothesis['supporting_evidences']:
                analysis_plan = evidence['analysis_plan']
                analyses.append(
                    {
                        "PMID":data["PMID"],
                        "analysis": analysis_plan
                    }
                )

    df_analysis = pd.DataFrame(analyses)

    if debug:
        df_analysis = df_analysis.iloc[0:10]

    # create the batch inputs
    batch_inputs = []
    for index, row in df_analysis.iterrows():
        batch_inputs.append({
            "analysis_text": row["analysis"]
        })

    # call the llm
    outputs = batch_call_llm_json_output(
        prompt_template=PROMPT,
        batch_inputs=batch_inputs,
        llm="gpt-4o-mini",
        temperature=0.0,
        batch_size=10,
    )

    # parse the outputs
    all_parsed = []
    for output in outputs:
        parsed = process_llm_output(output)
        all_parsed.append(parsed.get("class", ""))
    all_parsed = pd.DataFrame(all_parsed, columns=["class"])
    all_parsed = pd.concat([df_analysis[["PMID","analysis"]], all_parsed], axis=1)
    all_parsed.to_csv(os.path.join(output_dir, "analysis_classification.csv"), index=False)
    print(f"Saved to {os.path.join(output_dir, 'analysis_classification.csv')}")


if __name__ == "__main__":
    debug = False
    categorize_analysis(debug)