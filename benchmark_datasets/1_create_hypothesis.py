"""
This script creates a hypothesis for the benchmark dataset.

Load the abstract of the paper, get the main
hypothesis and the supporting evidences.
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

finding_extraction_prompt = """
The following is the abstract of a publication:

{abstract}

Task:
Given the abstract of a publication, your task is to extract binary hypotheses and their supporting evidences that can be tested through data analysis.

Requirements for hypotheses and evidences:
1. Each hypothesis must be testable using statistical analysis or machine learning methods
2. All evidence must include specific, measurable quantities or statistical relationships
3. Result values must be numerical (e.g., percentages, counts, p-values, correlation coefficients) or categorical with clear classifications
4. Analysis variables must be specific data columns or features that exist in the dataset

Return your answer as a JSON object in the following format:
```json
{{
    "hypotheses": [
        {{
            "hypothesis": a specific, binary hypothesis that can be tested statistically, from the abstract, the one which is considered to be true from the study,
            "wrong_hypothesis": make a random perturbation of the hypothesis so that it is a wrong hypothesis
            "supporting_evidences": [ \\ the evidences that support the alternative hypothesis
                {{
                    "analysis_plan": a brief analysis plan that can yield this evidence,
                    "evidence": specific statistical finding or measurement,
                    "analysis_variables": list of exact variables/features needed for analysis,
                    "result_variable": the specific metric or statistical measure used,
                    "result_variable_value": numerical value, statistical measure, or categorical outcome,
                }},
                ...
            ]
        }},
        ...
    ]
}}
```
"""

def parse_llm_hypothesis_output(output_text):
    """
    Parse the LLM output to extract hypotheses, handling both proper JSON and malformed output.
    
    Args:
        output_text (str): The raw output text from LLM
        
    Returns:
        list: List of hypothesis dictionaries
    """
    # First try parsing as proper JSON
    try:
        data = json.loads(output_text)
        return data.get('hypotheses', [])
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract hypotheses manually
        hypotheses = []
        
        # Find all hypothesis blocks using regex
        hypothesis_pattern = r'\{\s*"hypothesis":\s*"([^"]+)",\s*"wrong_hypothesis":\s*"([^"]+)",\s*"supporting_evidences":\s*\[(.*?)\]\s*\}'
        matches = re.finditer(hypothesis_pattern, output_text, re.DOTALL)
        
        for match in matches:
            hypothesis_text = match.group(1)
            wrong_hypothesis_text = match.group(2)
            evidences_text = match.group(3)
            
            # Parse supporting evidences
            evidence_pattern = r'\{\s*"analysis_plan":\s*"([^"]+)",\s*"evidence":\s*"([^"]+)",\s*"analysis_variables":\s*\[(.*?)\],\s*"result_variable":\s*"([^"]+)",\s*"result_variable_value":\s*"([^"]+)"\}'
            evidence_matches = re.finditer(evidence_pattern, evidences_text, re.DOTALL)
            
            supporting_evidences = []
            for ev_match in evidence_matches:
                evidence = {
                    "analysis_plan": ev_match.group(1),
                    "evidence": ev_match.group(2),
                    "analysis_variables": [var.strip(' "') for var in ev_match.group(3).split(',') if var.strip()],
                    "result_variable": ev_match.group(4),
                    "result_variable_value": ev_match.group(5),
                }
                supporting_evidences.append(evidence)
            
            hypothesis = {
                "hypothesis": hypothesis_text,
                "wrong_hypothesis": wrong_hypothesis_text,
                "supporting_evidences": supporting_evidences
            }
            hypotheses.append(hypothesis)
            
        return hypotheses

def process_llm_output(llm_output):
    hypotheses = parse_llm_hypothesis_output(llm_output)
    
    # Print parsed hypotheses for verification
    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n Hypothesis {i}: {hyp['hypothesis']}")
        print(f"\n Wrong Hypothesis {i}: {hyp['wrong_hypothesis']}")
        for j, evidence in enumerate(hyp['supporting_evidences'], 1):
            print(f"  Supporting Evidence {j}:")    
            print(f"    - Analysis Plan: {evidence['analysis_plan']}")
            print(f"    - Evidence: {evidence['evidence']}")
            print(f"    - Variables: {evidence['analysis_variables']}")
            print(f"    - Result Variable: {evidence['result_variable']}")
            print(f"    - Result Value: {evidence['result_variable_value']}")
    
    return hypotheses

def phase1():
    # load the study metadata
    df = pd.read_csv("/home/zifengw2/data/DSWizard/benchmark_datasets/cBioPortal/study_metadata.csv")
    output_dir = "/home/zifengw2/data/DSWizard/benchmark_datasets/cBioPortal/hypothesis"
    os.makedirs(output_dir, exist_ok=True)
    df_study_to_dataset = df[["PMID","dataset_id"]].groupby("PMID").agg(list).reset_index()
    df_study = df[["PMID","Title","Abstract","Results"]].drop_duplicates(subset=["PMID"])
    df_study.fillna("", inplace=True)
    study_data_list = []
    for index, row in df_study_to_dataset.iterrows():
        pmid = row["PMID"]
        dataset_ids = row["dataset_id"]
        title = df_study[df_study["PMID"] == pmid]["Title"].values[0]
        abstract = df_study[df_study["PMID"] == pmid]["Abstract"].values[0]
        results = df_study[df_study["PMID"] == pmid]["Results"].values[0]
        study_data_list.append(
            {
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract,
                "Results": results,
                "dataset_ids": dataset_ids
            }
        )

    # try batch call llm
    batch_inputs = []
    for study_data in study_data_list:
        title = study_data["Title"]
        abstract = study_data["Abstract"]
        if abstract is None:
            abstract = ""
        if title is None:
            title = ""
        
        abstract = title + "\n" + abstract
        batch_inputs.append({
            "abstract": abstract
        })

    # DEBUG
    # batch_inputs = batch_inputs[221:222]
    # batch_inputs = batch_inputs[222:223]
    # batch_inputs = batch_inputs[221:223]
    outputs = batch_call_llm_json_output(
        prompt_template=finding_extraction_prompt,
        batch_inputs=batch_inputs,
        llm="gpt-4o",
        temperature=0.0,
        batch_size=10,
        max_completion_tokens=2048,
    )

    # pase all the outputs
    all_parsed = []
    for output in outputs:
        parsed = process_llm_output(output)
        all_parsed.append(parsed)


    # save the parsed hypotheses with the dataset ids
    for parsed, study_data in zip(all_parsed, study_data_list):
        study_data["hypotheses"] = parsed
        output_file = os.path.join(output_dir, f"{study_data['PMID']}.json")
        with open(output_file, "w") as f:
            json.dump(study_data, f)


def phase2():
    # load all the parsed hypotheses
    # and do EDA
    # save the EDA results
    input_dir = "~/DSWizard/benchmark_datasets/cBioPortal/hypothesis"
    glob_pattern = os.path.join(input_dir, "*.json")
    all_files = glob(glob_pattern)
    all_hypotheses = []
    for file in all_files:
        with open(file, "r") as f:
            study_data = json.load(f)
            all_hypotheses.extend(study_data["hypotheses"])

    number_of_hypotheses = len(all_hypotheses)
    # number of evidences
    number_of_evidences = sum([len(hyp["supporting_evidences"]) for hyp in all_hypotheses])

def phase3():
    # TODO: consider creating data science coding question and answers from the result section
    pass

if __name__ == "__main__":
    phase1()
    phase2()