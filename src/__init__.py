import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
print("REPO_ROOT", REPO_ROOT)
sys.path.append(REPO_ROOT)
ENV_FILE = os.path.join(REPO_ROOT, ".env")
TOP_LEVEL_LOG_DIR = os.path.join(REPO_ROOT, "logs")

HYPOTHESIS_DIR = os.path.join(REPO_ROOT, "benchmark_datasets/cBioPortal/hypothesis")
