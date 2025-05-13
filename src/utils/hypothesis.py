import os
import json
import pickle
from pydantic import BaseModel
from typing import List, Dict, Union, Any
from .dataset import DatasetLoader

class Evidence(BaseModel):
    analysis_plan: str = ""
    evidence: str = ""
    analysis_variables: List[str]
    result_variable: str
    result_variable_value: Any
    
    def __repr__(self):
        return f"Evidence(\nanalysis_plan={self.analysis_plan}, \nevidence={self.evidence}, \nanalysis_variables={self.analysis_variables}, \nresult_variable={self.result_variable}, \nresult_variable_value={self.result_variable_value})"

class Hypothesis(BaseModel):
    hypothesis: str
    wrong_hypothesis: str = None # optional, in the case of non-verifiable hypotheses
    supporting_evidences: List[Evidence] = [] # optional, in the case of non-verifiable hypotheses
    
    def __repr__(self):
        return f"Hypothesis(\nhypothesis={self.hypothesis}, \nwrong_hypothesis={self.wrong_hypothesis}, \nsupporting_evidences={self.supporting_evidences})"

class HypothesisMetadata(BaseModel):
    PMID: int
    Title: str
    Abstract: str
    Results: str = None # optional, in the case of non-verifiable hypotheses
    dataset_ids: List[str]
    hypotheses: List[Hypothesis]
    
    def __repr__(self):
        return f"HypothesisMetadata(\nPMID={self.PMID}, \nTitle={self.Title}, \nAbstract={self.Abstract}, \nResults={self.Results}, \ndataset_ids={self.dataset_ids}, \nhypotheses={len(self.hypotheses)} hypotheses)"
    
class HypothesisLoader:

    def __init__(self, hypothesis_directory_path: str):
        self.hypothesis_directory_path = hypothesis_directory_path
        self.hypothesis_data: Dict[str, HypothesisMetadata] = {}
        self.load_hypothesis()

    def load_hypothesis(self):
        
        if (os.path.exists(self.hypothesis_directory_path)):
            for file in os.listdir(self.hypothesis_directory_path):
                if (file.endswith(".json")):
                    with open(os.path.join(self.hypothesis_directory_path, file), "r") as f:
                        json_data = json.load(f)
                        hypothesis = HypothesisMetadata(**json_data)
                        self.hypothesis_data[(str(hypothesis.PMID))] = hypothesis

        if (not os.path.exists(self.hypothesis_directory_path)):
            raise Exception("Hypothesis directory not found")
        
        self.min_range = 0
        self.max_range = len(self.hypothesis_data)

    def get_hypothesis(self, PMID: str, dataset_loader: Union[DatasetLoader, None] = None) -> HypothesisMetadata:
        
        # load the hypothesis data
        hypothesis = self.hypothesis_data[str(PMID)]

        # (optionally)load the datasets
        if (dataset_loader is not None):
            datasets = {}
            for dataset_id in hypothesis.dataset_ids:
                dataset = dataset_loader.get_dataset(dataset_id)
                datasets[dataset_id] = dataset
            
            return hypothesis, datasets
        
        return hypothesis, None
