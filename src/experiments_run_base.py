import dotenv
dotenv.load_dotenv()
import docker
import json
import os
from utils.dataset import DatasetLoader, get_tables_at_path
from utils.hypothesis import HypothesisLoader
from agents.coder_agent import CoderAgentV2
from agents.react_agent import ReActAgentV2
from agents.reasoning_coder_agent import ReasoningCoderAgent
from agents.reasoning_react_agent import ReasoningReactAgent
from tools.DockerSandbox.DockerSandboxTool import DockerSandboxTool
from tools.DockerSandbox.sandbox_core.ExecutionSandboxWrapper import EvalDataset
import logging
import traceback
import threading
from typing import List
import sys
import argparse
import hashlib
import psutil
import resource
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import (
    REPO_ROOT,
    TOP_LEVEL_LOG_DIR
)

num_threads = 1

# OPTIONALLY, IF YOU WANT TO RUN THIS MULTITHREADED, UNCOMMENT THE FOLLOWING LINES
# num_threads = psutil.cpu_count(logical=False)
# # Increase the system file descriptor limit
# # Get the current soft and hard limits
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# # Set the soft limit to the maximum allowed hard limit
# hard = int(hard * 0.9)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
# print(f"Increased file descriptor limit from {soft} to {hard}")

# Configure logger to output to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# location of the dataset to evaluate
BASE_DATASET_PATH = os.path.join(REPO_ROOT, "benchmark_datasets/cBioPortal")

# location of the hypothesis & dataset metadata
BASE_HYPOTHESIS_PATH = os.path.join(BASE_DATASET_PATH, "hypothesis")
BASE_DATASET_METADATA_PATH = os.path.join(BASE_DATASET_PATH, "dataset_metadata")

# constants for the  docker sandbox
PLACEHOLDER_PATH = "/workdir"
DATAHUB_PATH = os.path.join(BASE_DATASET_PATH, "datahub/public")

# for logger
LOG_DIR = os.path.join(TOP_LEVEL_LOG_DIR, "experiment_logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_experiment_log_pattern(pmid: str, hypothesis_index: int, is_true: bool, experiment_config: dict) -> str:
    """
    Generate the log file pattern for a specific experiment configuration.
    
    Args:
        pmid: Paper/experiment identifier
        hypothesis_index: Index of the hypothesis being tested
        is_true: Whether this is the true hypothesis or false hypothesis test
        experiment_config: Experiment configuration dictionary
    
    Returns:
        Path to the log file for this experiment
    """
    # Create naming attributes without sensitive data
    naming_attributes = experiment_config.copy()
    naming_attributes.pop("api_key", None)
    naming_attributes.pop("endpoint", None)
    naming_attributes.pop("api_type", None)
    config_substr = "|".join([f"{k}_{v}" for k, v in naming_attributes.items()])
    
    filename = f"experiment|{pmid}|hypothesis-{hypothesis_index}-{'true' if is_true else 'false'}|{config_substr}.json"
    
    # Check if filename is too long (max is typically 255 chars)
    if len(filename) > 255:
        # Hash the config string to create a shorter, fixed-length identifier
        config_hash = hashlib.md5(config_substr.encode('utf-8')).hexdigest()
        # Use a shorter prefix format with the hash
        filename = f"exp_{pmid}_h{hypothesis_index}{'t' if is_true else 'f'}_{config_hash}.json"
    
    return os.path.join(LOG_DIR, filename)

def run_experiment(
    hypothesis_loader: HypothesisLoader,
    dataset_loader: DatasetLoader,
    pmid: str, # id for the set of questions
    hypothesis_index: int, # id for the question
    experiment_config: dict, 
    agent_debug_mode: bool = False
):
    """
    Run a single experiment with both true and false hypotheses.
    
    Args:
        hypothesis_loader: Loader for hypothesis data
        dataset_loader: Loader for dataset metadata
        pmid: Paper/experiment identifier
        hypothesis_index: Index of the hypothesis being tested
        experiment_config: Experiment configuration dictionary
        agent_debug_mode: Whether to enable agent debugging
    """
    
    # collect the core hypothesis data
    hypotheses, dataset = hypothesis_loader.get_hypothesis(pmid, dataset_loader)
    h_data = hypotheses.hypotheses[hypothesis_index]
    
    # create a copy of the experiment config and remove the api_key and endpoint
    cleaned_experiment_config = experiment_config.copy()
    cleaned_experiment_config.pop("api_key")
    cleaned_experiment_config.pop("endpoint")
    
    # add additional metadata to the result
    cleaned_experiment_config['pmid'] = pmid
    cleaned_experiment_config['hypothesis_index'] = hypothesis_index
    cleaned_experiment_config['dataset_ids'] = hypotheses.dataset_ids
    
    if (
        os.path.exists(
            get_experiment_log_pattern(pmid, hypothesis_index, True, cleaned_experiment_config)
        ) and os.path.exists(
            get_experiment_log_pattern(pmid, hypothesis_index, False, cleaned_experiment_config)
        )
    ):
        logger.info(f"Experiment log files already exist: {experiment_config['agent_type']} - {experiment_config['agent_type']} - {pmid} - {hypothesis_index}")
        return
    
    
    # initialize the docker sandbox
    sandbox = DockerSandboxTool(
        clip_output=10000
    )
        
    logger.info(f"Running new experiment for PMID {pmid}, hypothesis {hypothesis_index}, config: {experiment_config.get('agent_type')} - {experiment_config.get('model_name', '')}")
    
    # initialize the docker sandbox with the current dataset:
    table_paths_SANDBOX = []
    table_paths_LOCAL = []
    for dataset_id in hypotheses.dataset_ids:
        dataset = dataset_loader.get_dataset(dataset_id)
        table_paths_SANDBOX.extend(get_tables_at_path(dataset, PLACEHOLDER_PATH))
        table_paths_LOCAL.extend(get_tables_at_path(dataset, DATAHUB_PATH))
        
    # load the dataset into memory, and track the mapping between local and sandbox paths
    dataset_for_sandbox = EvalDataset(
        local_table_paths=table_paths_LOCAL,
        target_table_paths=table_paths_SANDBOX,
    )
    
    # upload the tables to the docker sandbox
    # if they are already uploaded, the upload will be skipped
    sandbox.upload_tables(dataset=dataset_for_sandbox)
    
    logger.info("experiment config:")
    for k, v in experiment_config.items():
        logger.info(f"\t{k}: {v}")
    
    logger.debug(f"\tStated Hypothesis: {h_data.hypothesis}")
    logger.debug(f"\tNegative (wrong) Hypothesis: {h_data.wrong_hypothesis}")
    
    
    agent_type = experiment_config["agent_type"]
    
    agent_invoke = None
    
    if agent_type == "react":
        
        agent = ReActAgentV2(
            model_name=experiment_config["model_name"],
            final_response_model=experiment_config["final_response_model"],
            api_type=experiment_config["api_type"],
            api_key=experiment_config["api_key"],
            endpoint=experiment_config["endpoint"],
            language=experiment_config["language"]
        ).withTools(
            tools=[sandbox]
        ).withDefaultSystemPrompt(
            query=h_data.hypothesis,
            dataset_paths="\n".join(table_paths_SANDBOX),
            dataset_schema=dataset.simple_schema_description
        ).create_agent_graph(
            debug=agent_debug_mode
        )
        
        step_count = experiment_config['step_count']
        agent_invoke = lambda input_query: agent.generate(input_query=input_query, remaining_steps=step_count)

    elif agent_type == "coder":
        agent = CoderAgentV2(
            model_name=experiment_config["model_name"],
            final_response_model=experiment_config["final_response_model"],
            api_type=experiment_config["api_type"],
            api_key=experiment_config["api_key"],
            endpoint=experiment_config["endpoint"],
            language=experiment_config["language"]
        ).withTools(
            tools=[sandbox]
        ).withDefaultSystemPrompt(
            dataset_paths="\n".join(table_paths_SANDBOX),
            dataset_schema=dataset.simple_schema_description
        ).create_agent_graph(
            debug=agent_debug_mode
        )
        
        agent_invoke = lambda input_query: agent.generate(input_query=input_query)
    
    elif (agent_type == "reasoning_coder"):
        agent = ReasoningCoderAgent(
            api_type=experiment_config["api_type"],
            api_key=experiment_config["api_key"],
            endpoint=experiment_config["endpoint"],
            language=experiment_config["language"],
            planning_model=experiment_config["planning_model"],
            coding_model=experiment_config["coding_model"],
            final_response_model=experiment_config["final_response_model"]
        ).withTools(
            tools=[sandbox]
        ).withDefaultSystemPrompt(
            dataset_paths="\n".join(table_paths_SANDBOX),
            dataset_schema=dataset.simple_schema_description
        ).create_agent_graph(
            debug=agent_debug_mode
        )
        
        agent_invoke = lambda input_query: agent.generate(input_query=input_query)
            
    elif agent_type == "reasoning_react_v2":
        agent = ReasoningReactAgent(
            plan_model_name=experiment_config["plan_model_name"],
            agent_model_name=experiment_config["agent_model_name"],
            final_response_model=experiment_config["final_response_model"],
            api_type=experiment_config["api_type"],
            api_key=experiment_config["api_key"],
            endpoint=experiment_config["endpoint"],
            language=experiment_config["language"]
        ).withTools(
            tools=[sandbox]
        ).withDefaultSystemPrompt(
            query=h_data.hypothesis,
            dataset_paths="\n".join(table_paths_SANDBOX),
            dataset_schema=dataset.simple_schema_description
        ).create_agent_graph(
            debug=agent_debug_mode
        )
        
        step_count = experiment_config['step_count']
        agent_invoke = lambda input_query: agent.generate(input_query=input_query, remaining_steps=step_count)

    else:
        raise ValueError(f"Agent type {agent_type} is not supported")
    
    # invoke the agent with the base hypothesis
    result = agent_invoke(h_data.hypothesis)
    result['experiment_config'] = cleaned_experiment_config
    result['hypothesis_is_true'] = True
    
    # write the result to a file
    log_file = get_experiment_log_pattern(pmid, hypothesis_index, True, cleaned_experiment_config)
    with open(log_file, "w") as f:
        json.dump(result, f)
        
    # clear the workspace before the next experiment
    sandbox.sandbox.clear_workspace()
    
    # invoke the agent with the negative (synthetically generated) hypothesis
    result = agent_invoke(h_data.wrong_hypothesis)
    result['experiment_config'] = cleaned_experiment_config
    result['hypothesis_is_true'] = False
    log_file = get_experiment_log_pattern(pmid, hypothesis_index, False, cleaned_experiment_config)
    with open(log_file, "w") as f:
        json.dump(result, f)

    # clear the workspace before the next experiment
    sandbox.sandbox.clear_workspace()
    
    try:
        sandbox.sandbox.stop()
    except Exception as e:
        print(e)
        
    logger.debug(f"Experiment completed")
    

def get_coder_experiments() -> List[dict]:
    """
    Create and return a list of coder agent experiment configurations.
    
    Returns:
        List of experiment configuration dictionaries
    """
    
    experiments = []

    # create the coder experiments
    base_coder_experiment_config = {
        "agent_type": "coder",
        "api_type": "azure",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "language": "python",
        "final_response_model": "gpt-4o"
    }
    for model_name in ["gpt-4o", "o3-mini"]:
        experiment = base_coder_experiment_config.copy()
        experiment["model_name"] = model_name
        experiments.append(experiment)
        
    with open("coder_experiments.json", "w") as f:
        json.dump(experiments, f)

    return experiments

def get_react_experiments() -> List[dict]:
    """
    Create and return a list of ReAct agent experiment configurations.
    
    Returns:
        List of experiment configuration dictionaries
    """
    experiments = []
    
    # create the react experiments
    base_react_experiment_config = {
        "agent_type": "react",
        "api_type": "azure",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "language": "python",
        "final_response_model": "gpt-4o"
    }
    
    for model_name in ["gpt-4o", "o3-mini"]:
        experiment = base_react_experiment_config.copy()
        experiment["model_name"] = model_name
        for step_count in [4, 8, 16]:
            experiment["step_count"] = step_count
            experiments.append(experiment)
            
    with open("react_experiments.json", "w") as f:
        json.dump(experiments, f)
            
    return experiments

def get_reasoning_coder_experiments() -> List[dict]:
    """
    Create and return a list of reasoning coder agent experiment configurations.
    
    Returns:
        List of experiment configuration dictionaries
    """
    experiments = []
    
    # create the reasoning coder experiments
    base_reasoning_coder_experiment_config = {
        "agent_type": "reasoning_coder",
        "api_type": "azure",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "language": "python",
        "final_response_model": "gpt-4o"
    }
    for model_name in ["gpt-4o"]:
        experiment = base_reasoning_coder_experiment_config.copy()
        experiment["model_name"] = model_name
        for planning_model in ["o3-mini"]:
            experiment["planning_model"] = planning_model
            
            for coding_model in ["gpt-4o"]:
                experiment["coding_model"] = coding_model
                experiments.append(experiment)
                
    with open("reasoning_coder_experiments.json", "w") as f:
        json.dump(experiments, f)
                
    return experiments

def get_reasoning_react_experiments() -> List[dict]:
    """
    Create and return a list of reasoning react agent experiment configurations.
    
    Returns:
        List of experiment configuration dictionaries
    """
    experiments = []
    
    # create the reasoning react experiments
    base_reasoning_react_experiment_config = {
        "agent_type": "reasoning_react_v2",
        "plan_model_name": "gpt-4o",
        "agent_model_name": "gpt-4o",
        "final_response_model": "gpt-4o",
        "api_type": "azure",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "language": "python"
    }
    
    for plan_model_name in ["gpt-4o", "o3-mini"]:
        experiment = base_reasoning_react_experiment_config.copy()
        experiment["plan_model_name"] = plan_model_name
        for agent_model_name in ["gpt-4o", "o3-mini"]:
            experiment["agent_model_name"] = agent_model_name
            for step_count in [4, 8, 16]:
                experiment["step_count"] = step_count
                experiments.append(experiment)
                
    with open("reasoning_react_experiments.json", "w") as f:
        json.dump(experiments, f)
                
    return experiments

if __name__ == "__main__":

    # the main experiment loop

    # load the dataset
    dataset_loader = DatasetLoader(
        dataset_metadata_directory_path=BASE_DATASET_METADATA_PATH,
    )

    # load the hypothesis
    hypothesis_loader = HypothesisLoader(
        hypothesis_directory_path=BASE_HYPOTHESIS_PATH,
    )
    pmid_list = list(hypothesis_loader.hypothesis_data.keys())
    
    # load the experiments
    experiments = []
    experiments.extend(get_coder_experiments())
    experiments.extend(get_react_experiments())
    experiments.extend(get_reasoning_coder_experiments())
    experiments.extend(get_reasoning_react_experiments())
    
    print(f"Running {len(experiments)} experiment configurations")
    
    file_writing_lock = threading.Lock()
    
    # used to write the logs for each experiment
    work_level_logs_file = os.path.join(LOG_DIR, "work_level_logs.txt")
    def write_work_level_log(log_str):
        with file_writing_lock:
            with open(work_level_logs_file, "a") as f:
                f.write(log_str + "\n")
    
    # used to write the timing logs for each experiment
    timing_log_file_lock = threading.Lock()
    timing_log_file = os.path.join(LOG_DIR, "timing_log.txt")
    def write_timing_log(log_str):
        with timing_log_file_lock:
            with open(timing_log_file, "a") as f:
                f.write(log_str + "\n")
    
    # used to run the unit of work for each experiment
    def run_unit_of_work(pmid, hypothesis_index, experiment):
        
        log_str = f"PMID: {pmid}, Hypothesis Index: {hypothesis_index}, Experiment Config: {experiment}"
        try:
            start_time = time.time()
            run_experiment(
                hypothesis_loader=hypothesis_loader,
                dataset_loader=dataset_loader,
                pmid=pmid,
                hypothesis_index=hypothesis_index,
                experiment_config=experiment,
                agent_debug_mode=False   
            )
            log_str += "\nSUCCESS"
            end_time = time.time()
            run_time = end_time - start_time
            # Format runtime in seconds with 2 decimal places
            run_time_str = f"{run_time:.2f}"
            write_timing_log(run_time_str)
        except Exception as e:    
            log_str += f"\nERROR IN EXPERIMENT: {e} {traceback.format_exc()}"
            client = docker.from_env()
            
            try:
                client.containers.prune()
            except Exception as e:
                log_str += f"\nERROR PRUNING CONTAINERS: {e} {traceback.format_exc()}"
                
        finally:
            write_work_level_log(log_str)

    from queue import Queue
    
    work_queue = Queue()
    
    def shuffle_iter():
        for pmid in pmid_list:
            hypothesis_data, _ = hypothesis_loader.get_hypothesis(pmid)
            for hypothesis_index in range(len(hypothesis_data.hypotheses)):
                yield (pmid, hypothesis_index)
                
    # build the work queue
    for experiment in experiments:        
        for pmid, hypothesis_index in shuffle_iter():
            work_queue.put((pmid, hypothesis_index, experiment))

    print(f"Running {num_threads} threads")
    threads = []
    
    # series of threads work through the work queue. Each thread will finish its unit of work and pick up the next unit of work.
    def worker():
        while not work_queue.empty():
            try:
                pmid, hypothesis_index, experiment = work_queue.get()
                run_unit_of_work(pmid, hypothesis_index, experiment)
                work_queue.task_done()
            except Exception as e:
                write_work_level_log(f"ERROR IN WORKER THREAD: {e} {traceback.format_exc()}")
            
    # Create and start the worker threads
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        thread.daemon = True  # Set as daemon so they exit when the main thread exits
        threads.append(thread)
        thread.start()
    
    # Wait for all work to be completed
    work_queue.join()
    
    print(f"All experiments completed. Check {work_level_logs_file} for detailed logs.")