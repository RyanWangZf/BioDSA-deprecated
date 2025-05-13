from glob import glob
import os
import json
import pdb
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, List
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import (
    REPO_ROOT,
    TOP_LEVEL_LOG_DIR,
    HYPOTHESIS_DIR
)
from src.evaluate import llm_evaluate_evidence_alignment
from src.utils.hypothesis import HypothesisLoader

LOG_DIR = os.path.join(TOP_LEVEL_LOG_DIR, "experiment_logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

OUTPUT_DIR = os.path.join(TOP_LEVEL_LOG_DIR, "eval_evidence_alignment")
COMPLETED_TASKS_FILE = os.path.join(OUTPUT_DIR, "completed_tasks.txt")
EVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "eval_results.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# load the groundtruth hypothesis
hypothesis_loader = HypothesisLoader(
    hypothesis_directory_path=HYPOTHESIS_DIR
)

# extract the agent type, which is composed of the agent_type, and it's hyperparameters
def get_agent_type(row: Dict) -> List[str]:
    """
    Format agent type based on configuration.
    For react agent: (react, step_count)
    For reasoning coder: (reasoning_coder, planning_model, coding_model)
    For coder: (coder, model_name)
    """
    agent_type = row["agent_type"]
    hyperparameters = []
    if agent_type == "react":
        hyperparameters = [
            str(row['step_count']), 
            row['model_name']
        ]
    elif agent_type == "reasoning_coder":
        hyperparameters = [
            row['planning_model'], 
            row['coding_model']
        ]
    elif agent_type == "coder":
        hyperparameters = [
            row['model_name']
        ]
    elif agent_type == "reasoning_coder_with_react":
        hyperparameters = [
            row['plan_model_name'], 
            row['agent_model_name'],
            row['step_count']
        ]
    return [agent_type] + hyperparameters

def _get_gt_evidence(pmid: str, hypothesis_index: int):
    gt_hypothesis, _ = hypothesis_loader.get_hypothesis(pmid)
    gt_hypothesis = gt_hypothesis.hypotheses[hypothesis_index]
    gt_evidence = gt_hypothesis.supporting_evidences
    gt_evidence_list = []
    for evidence in gt_evidence:
        gt_evidence_list.append(evidence.evidence)  
    return gt_evidence_list

# load the logs
logs = glob(os.path.join(LOG_DIR, "*.json"))

"""
This file contains lines of the task_id that have been completed.
"""
# load the completed tasks
completed_tasks = []
if (os.path.exists(COMPLETED_TASKS_FILE)):
    with open(COMPLETED_TASKS_FILE, "r") as f:
        completed_tasks = f.readlines()
    
    completed_tasks = [line.strip() for line in completed_tasks]

completed_tasks = set(completed_tasks)

eval_results = []
for log in tqdm(logs):
    try:
        with open(log, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"WARNING: Error loading {log}: {e}")
        continue
    
    pmid = data['experiment_config']['pmid']
    hypothesis_index = data['experiment_config']['hypothesis_index']
    gt_evidences = _get_gt_evidence(pmid, hypothesis_index)
    
    # get the task id string
    agent_type = get_agent_type(data["experiment_config"])
    task_id_string = "|".join([
        pmid,
        str(hypothesis_index),
        str(data['hypothesis_is_true']),
        "|".join(sorted(data['experiment_config']['dataset_ids'])),
        *agent_type
    ]).strip()
    if task_id_string in completed_tasks:
        print(f"Skipping {task_id_string} because it has already been completed")
        continue
    
    generated_evidences = data["evidence"]
    eval_alignment_list = llm_evaluate_evidence_alignment(gt_evidences=gt_evidences, generated_evidences=generated_evidences)
    save_result = {
        "experiment_config": data["experiment_config"],
        "ground_truth_evidence": gt_evidences,
        "generated_evidence": generated_evidences,
        "eval_evidence_alignment": eval_alignment_list
    }
    eval_results.append(save_result)
    completed_tasks.add(task_id_string)
    with open(COMPLETED_TASKS_FILE, "a") as f:
        f.write(task_id_string + "\n")

# Load existing results if file exists
existing_results = []
results_file = os.path.join(OUTPUT_DIR, "eval_results.json")
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: Could not load existing results from {results_file}")

# Append new results
all_results = existing_results + eval_results

# Write back all results
with open(EVAL_RESULTS_FILE, "w") as f:
    json.dump(eval_results, f, indent=4)
