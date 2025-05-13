import json
import os
import pickle
from pydantic import BaseModel
from typing import List, Dict, Any

"""
Types that define the dataset metadata
"""
class ColumnMetadata(BaseModel):
    name: str
    data_type: str
    statistics: Dict[str, Any]
    n_unique: int
    missing_rate: float

class TableMetadata(BaseModel):
    name: str
    description: str
    columns: List[ColumnMetadata]
    
class DatasetMetadata(BaseModel):
    dataset_id: str
    type_of_cancer: str
    description: str
    tables: List[TableMetadata]
    simple_schema_description: str # used in prompting agents

def get_tables_at_path(dataset: DatasetMetadata, base_path: str) -> DatasetMetadata:
    """
    Utility function to get the path list for the tables in the dataset
    """
    tables = []
    for table in dataset.tables:
        table_path = os.path.join(os.path.join(base_path, dataset.dataset_id), table.name)
        tables.append(table_path)
    return tables

def create_simple_schema_description(json_data: Dict[str, Any]) -> str:
    """
    Create a simple schema description for the dataset
    """
    tables = json_data["tables"]
    simple_schema_description = ""
    for table in tables:
        # if there are comment rows, add a note
        simple_schema_description += f"Table: {table['name']}\n"
        n_comment_rows = table.get("n_comment_rows", 0)
        comment_rows = table.get("comment_rows", [])
        if n_comment_rows > 0:
            simple_schema_description += f"Note: the first {n_comment_rows} rows of the table are comment rows starting with '#':\n"
            comment_rows = "\n".join(comment_rows)
            simple_schema_description += f"{comment_rows}\n"
        simple_schema_description += f"Note: this table is tab-delimited.\n"
        # only take the first 20 columns' names and data types and five example values
        if len(table["columns"]) > 20:
            columns = table["columns"][:20]
        else:
            columns = table["columns"]
        simple_schema_description += f"Columns:\n"
        for column in columns:
            stats = column["statistics"]
            stats_type = stats.pop("statistics_type")
            # convert the dict stats to a string
            stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
            simple_schema_description += f"- `{column['name']}`: {column['data_type']} ({stats_type}: {stats_str})\n"
        # if there are more than 20 columns, add a note
        if len(table["columns"]) > 20:
            simple_schema_description += f"[...] Note: Only the first 20 columns are shown, the rest {len(table["columns"]) - 20} columns are not shown here.\n"
        simple_schema_description += f"-----------------------------------\n"
    return simple_schema_description

class DatasetLoader:

    def __init__(
        self,
        dataset_metadata_directory_path: str,
    ):
        """
        Initialize the dataset loader

        Args:
            dataset_metadata_directory_path: The path to the directory containing the dataset metadata
            use_cache: Whether to use a cache file to store the dataset metadata
            cache_file_path: The path to the cache file
        """
        self.dataset_metadata_directory_path = dataset_metadata_directory_path
        self.dataset_metadata: Dict[str, DatasetMetadata] = {}
        self.load_dataset_metadata()
    
    def load_dataset_metadata(self):

        if (not os.path.exists(self.dataset_metadata_directory_path)):
            raise Exception("Dataset metadata directory not found at: " + self.dataset_metadata_directory_path)
    
        for file in os.listdir(self.dataset_metadata_directory_path):            
            if (file.endswith(".json")):
                with open(os.path.join(self.dataset_metadata_directory_path, file), "r") as f:
                    json_data = json.load(f)
                    # create the example head five rows of the dataset
                    simple_schema_description = create_simple_schema_description(json_data)
                    json_data["simple_schema_description"] = simple_schema_description
                    dataset_metadata = DatasetMetadata(**json_data)
                    self.dataset_metadata[dataset_metadata.dataset_id] = dataset_metadata

    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        return self.dataset_metadata[dataset_id]


