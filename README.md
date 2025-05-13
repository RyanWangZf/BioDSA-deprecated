# BIODSA-1K: Benchmarking Data Science Agents for Biomedical Research

## Abstract
Validating scientific hypotheses is a central challenge in biomedical research, and remains difficult for artificial intelligence (AI) agents due to the complexity of real-world data analysis and evidence interpretation. In this work, we present BIOPDDSA-1K, a benchmark designed to evaluate AI agents on realistic, data-driven biomedical hypothesis validation tasks. BIOPDDSA-1K consists of 1,029 hypothesis-centric tasks paired with 1,177 analysis plans, curated from over 300 published biomedical studies to reflect the structure and reasoning found in authentic research workflows. Each task includes a structured hypothesis derived from the original study's conclusions, expressed in the affirmative to reflect the language of scientific reporting, and one or more pieces of supporting evidence grounded in empirical data tables. While these hypotheses mirror published claims, they remain testable using standard statistical or machine learning methods. The benchmark enables evaluation along four axes: (1) hypothesis decision accuracy, (2) alignment between evidence and conclusion, (3) correctness of the reasoning process, and (4) executability of the AI-generated analysis code. Importantly, BIOPDDSA-1K includes non-verifiable hypotheses: cases where the available data are insufficient to support or refute a claim, reflecting a common yet underexplored scenario in real-world science. We propose BIOPDDSA-1K as a foundation for building and evaluating generalizable, trustworthy AI agents for biomedical discovery.

## Benchmark data

The benchmark data is available at: https://huggingface.co/datasets/zifeng-ai/BioDSA-1K.


## About This Repository
The repository contains the code and data for the paper "BIODSA-1K: Benchmarking Data Science Agents for Biomedical Research". The repository is organized into several directories:
- `benchmark_datasets` contains the code to generate the benchmark datasets from scratch, and also serves as the location for the benchmark datasets.
- `src` contains the code for experiments, and all supporting functions/classes to facilitate the experiments.
- `experiment_visualization` contains the code to visualize the results of the experiments.

Additionally, there are some crucial files in the root directory:
- `README.md`: This file.
- `Pipfile`: The requirements for the project.
- `Pipfile.lock`: The locked requirements for the project. (for reproducibility)
- `.env`: The environment file for the project. (for reproducibility)

## Requirements
- Python 3.12 (enforced by `pipenv`)
- docker engine (for running the experiments)
- Ubuntu 20.04 or later (for running the experiments), or MacOS Ventura or later. 

There are many ways to install docker, but the easiest way is to use the official installation script. You can find the script [here](https://get.docker.com/). The script is a simple bash script that installs docker on your machine. You can run it with the following command:
```bash
#!/bin/bash
# Update the package repository
sudo apt-get update -y

# Install Docker
sudo apt-get install -y containerd
sudo apt-get install -y docker.io

# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker
```

## Installation Steps
0. install `pipenv` if you don't have it already: https://pipenv.pypa.io/en/latest/
1. Clone the repository:
```bash
git clone {url}
cd {repo_name}
```
2. Create a virtual environment with dependencies:
```bash
pipenv install
```
In subsequent steps, you can use `pipenv run {your command + args here}` to run the code in the virtual environment.

### Build the docker sandbox:
```bash
cd src/tools/DockerSandbox/sandbox_core/docker_container/build_sandbox.sh
./bash build_sandbox.sh
```
>Note: this does not need to be run in the Pipenv virtual environment. This simply creates the docker image invoked during experiments.

## Running the Experiments

Before running experiments double check:
a. the `docker` daemon is running. You can check this by running the following command:
```bash
docker info
```
b. The `docker` image is built. You can check this by running the following command:
```bash
docker images
```
The image `combined-sandbox` should be present. If it is not, run the build script in the previous section.
c. The variables `REPO_BASE_PATH`, `BASE_HYPOTHESIS_PATH`, `BASE_DATASET_METADATA_PATH`, `DATAHUB_PATH`, and `LOG_DIR` are correctly set. 
d. the `.env` file is correctly set. The `.env` file should contain the following variables:
```bash
AZURE_OPENAI_ENDPOINT="your endpoint"
AZURE_OPENAI_API_KEY="your endpoint key"
```


#### Run the base experiments:
```bash
pipenv run python src/experiments_run_base.py
```
#### Once experimental logs are generated, you can run the evidence alignment evaluation:
```bash
pipenv run python src/experiments_run_evidence_alignment.py
```
#### To run the non-verifiable hypothesis evaluation:
```bash
pipenv run python src/experiments_run_non_verifiable.py
```

# Reference

If you find this project useful, please consider citing the following paper:

```bibtext
@article{wang2025biodsa,
title = {BioDSA-1K: Benchmarking Data Science Agents for Biomedical Research},
author = {Wang, Zifeng and Danek, Benjamin and Sun, Jimeng},
year = {2025}
}
```
