"""
This script processes the cBioPortal dataset to create a benchmark dataset for the DSWizard project.
"""

import pandas as pd
import os
import re
import pdb
from glob import glob
import string
from tqdm import tqdm
import json
import traceback
cbioportal_dir = "~/cBioPortalData/datahub/public" # cBioPortal source directory
output_dir = "~/DSWizard/benchmark_datasets/cBioPortal" # output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import time

def _fetch_pubmed_data(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),  # Join PMIDs with commas
        "retmode": "xml"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    # Parse the XML response
    root = ET.fromstring(response.text)
    articles = []
    
    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text
        # Handle multiple abstract sections
        abstract_elems = article.findall(".//Abstract/AbstractText")
        if abstract_elems:
            # Combine all abstract sections with proper spacing
            abstract = " ".join(''.join(elem.itertext()) for elem in abstract_elems if elem is not None)
        else:
            abstract = "No abstract available"
        
        articles.append({"PMID": pmid, "Title": title, "Abstract": abstract})
    
    return articles

def _fetch_single_pmc(pmid: str) -> Tuple[str, Optional[Dict]]:
    """
    Fetch data for a single PMC ID with retry logic
    
    Returns:
        Tuple[str, Optional[Dict]]: (pmid, data) tuple where data is None if fetch failed
    """
    base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"
    url = base_url.format(pmid=pmid)
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return (pmid, response.json())
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to fetch PMID {pmid} after {max_retries} attempts: {str(e)}")
                return (pmid, None)
            time.sleep(retry_delay * (attempt + 1))

def _fetch_pmc_data(pmids: List[str], max_threads: int = 5) -> List[Tuple[str, Dict]]:
    """
    Fetch PMC data for multiple PMIDs using multiple threads
    
    Args:
        pmids (List[str]): List of PMIDs to fetch
        max_threads (int): Maximum number of concurrent threads (default: 5)
    
    Returns:
        List[Tuple[str, Dict]]: List of (pmid, data) tuples for successful fetches
    """
    results = []
    max_threads = min(max_threads, len(pmids))
    
    print(f"Fetching {len(pmids)} PMIDs using {max_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_pmid = {executor.submit(_fetch_single_pmc, pmid): pmid for pmid in pmids}
        
        for future in as_completed(future_to_pmid):
            pmid, data = future.result()
            if data is not None:
                results.append((pmid, data))
                print(f"[✓] PMID {pmid}")
            else:
                print(f"[✗] PMID {pmid}")
    
    # Sort results by original PMID order
    results.sort(key=lambda x: pmids.index(x[0]))
    
    print(f"\nFetch complete: {len(results)}/{len(pmids)} successful")
    
    return results

def phase1():
    """
    Get the study metadata for all the datasets in cBioPortal.
    """
    # get the list of datasets
    datasets = glob(os.path.join(cbioportal_dir, "*"))
    print(len(datasets))
    # for each dataset, get the study metadata
    all_meta_data = []
    for dataset in datasets:
        meta_datafile = os.path.join(dataset, "meta_study.txt")
        # read the txt file, it is in the format, each line is a key-value pair
        meta_data = {}
        with open(meta_datafile, "r") as f:
            for line in f:
                try:
                    line = line.strip()
                    if line == "":
                        continue
                    # only split on the first colon
                    key, value = line.strip().split(":", 1)
                    value = value.strip()
                    key = key.strip()
                    meta_data[key] = value
                except:
                    pdb.set_trace()
                    pass
            meta_data["dataset_id"] = os.path.basename(dataset)
        all_meta_data.append(meta_data)
    all_meta_data = pd.DataFrame(all_meta_data)
    all_meta_data.to_csv(os.path.join(output_dir, "study_metadata.csv"), index=False)

def phase2():
    """Get the pubmed publication for the datasets in cBioPortal.
    """
    # for pmids to get the publication title and abstract
    pubmed_api_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    # read the study metadata
    study_metadata = pd.read_csv(os.path.join(output_dir, "study_metadata.csv"))
    # get the pmids
    study_metadata = study_metadata.dropna(subset=["pmid"])
    # explode the data with pmid by ","
    study_metadata["pmid"] = study_metadata["pmid"].str.split(",")
    # only keep the rows with one pmid
    # cuz some with multiple pmids may cause the issue
    # that one paper may analyze multiple datasets
    # and one dataset can not be used to lead to the conclusion of the paper
    study_metadata = study_metadata[study_metadata["pmid"].map(len) == 1]
    study_metadata = study_metadata.explode("pmid").reset_index(drop=True)
    duplicate_pmids = study_metadata[study_metadata.duplicated(subset=['pmid'], keep=False)]
    print(duplicate_pmids)
    pmid_list = study_metadata["pmid"].unique().tolist()
    # get the publication title and abstract in csv format
    pubmed_data = _fetch_pubmed_data(pmid_list)
    pubmed_data = pd.DataFrame(pubmed_data)
    pubmed_data.to_csv(os.path.join(output_dir, "pubmed_data.csv"), index=False)
    # merge the pubmed data with the study metadata
    study_metadata.rename(columns={"pmid": "PMID"}, inplace=True)
    study_metadata = study_metadata.merge(pubmed_data, on="PMID", how="left")
    study_metadata.to_csv(os.path.join(output_dir, "study_metadata.csv"), index=False)

def _infer_data_type(X, column):
    # if the column contains any form of identifier, set it to be "categorical"
    # use regex to check if the column contains any form of identifier
    if re.search(r"(id|name|index|identifier|symbol|type|status|phase|source|stage)", column, re.IGNORECASE):
        return "categorical"

    ts = X[column].dropna().unique()
    # try to see if only two unique values are present
    if len(ts) == 2:
        return "binary"

    # try to see if all values are numbers
    try:
        numeric_ts = pd.to_numeric(ts)
        
        # Check if all numbers are integers by comparing with rounded values
        # This avoids the runtime warning from astype(int)
        if all(numeric_ts == numeric_ts.round()):
            return "integer"
        else:
            # If not all integers, it's continuous
            return "continuous"
    except:
        return "categorical"

def _build_statistics(X, column, data_type):
    n_unique = len(X[column].unique())
    if data_type == "binary":
       # convert all to string first
       ts = X[column]
       ts = ts.fillna("NA")
       ts = ts.astype(str)
       # count the number of unique values
       n_unique = len(ts.unique())
       # missing is either "NA" or ""
       missing_rate = round(((ts == "NA") | (ts == "")).mean(), 2)
       stats = ts.value_counts().to_dict()
       stats['statistics_type'] = "value_counts"
       return missing_rate, n_unique, stats
    elif data_type == "integer":
        # dropna first
        # convert to numeric first
        # and drop the non-numeric values
        ts = X[column]
        ts = pd.to_numeric(ts, errors="coerce")
        missing_rate = round(ts.isna().mean(), 2)
        ts = ts.dropna()
        quantiles = ts.quantile([0.01, 0.2, 0.4, 0.6, 0.8, 0.99]).to_dict()
        quantiles["min"] = ts.min()
        quantiles["max"] = ts.max()
        # make all values in quantiles to be integers
        for key in quantiles:
            # if the value is a NaN, set it to be "NA"
            if pd.isna(quantiles[key]):
                quantiles[key] = "NA"
            else:
                quantiles[key] = int(quantiles[key])
        quantiles['statistics_type'] = "quantiles"
        return missing_rate, n_unique, quantiles

    elif data_type == "continuous":
        # dropna first
        # convert to numeric first
        # and drop the non-numeric values
        ts = X[column]
        ts = pd.to_numeric(ts, errors="coerce")
        missing_rate = round(ts.isna().mean(), 2)
        ts = ts.dropna()
        desc = ts.describe().to_dict()
        # make all values to be within limited digits so that it is more readable
        for key in desc:
            # if the value is a NaN, set it to be ""
            # otherwise, if the value is a float, set it to be digits < 5
            if pd.isna(desc[key]):
                desc[key] = "NA"
            elif isinstance(desc[key], float):
                desc[key] = round(desc[key], 4)
            else:
                pass
        desc['statistics_type'] = "descriptive"
        return missing_rate, n_unique, desc
    elif data_type == "categorical":
        # try to convert to string first
        # if NaN or others, set it to be ""
        ts = X[column]
        ts = ts.fillna("")
        ts = ts.astype(str)
        # count missing values by the length == 0
        missing_rate = round((ts.str.len() == 0).mean(), 2)
        # count the number of unique values
        n_unique = len(ts.unique())
        # count the top 10 frequent values
        top_10_values = ts.value_counts().head(10).to_dict()
        top_10_values['statistics_type'] = "value_counts"
        return missing_rate, n_unique, top_10_values

def _column_names_clean(column_names):
    cleaned_column_names = []
    for idx, column_name in enumerate(column_names):
        try:
            column_name = str(column_name)
            column_name = column_name.replace(" ", "_")
            column_name = column_name.translate(str.maketrans('', '', string.punctuation))
            cleaned_column_names.append(column_name)
        except:
            cleaned_column_names.append(f"column_{idx}")
    # make the duplicates ones unique by adding "_" and a number
    cleaned_column_names = [
        f"{name}_{i}" if cleaned_column_names.count(name) > 1 else name
        for i, name in enumerate(cleaned_column_names)
    ]
    return cleaned_column_names

def _build_columns_metadata(data_file):
    # read the text file
    # skip the line starting with "#"
    all_values = []
    column_names = []
    comment_lines = 0
    comment_rows = []
    with open(data_file, "r") as f:
        line_idx = 0
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # skip the line starting with "#"
                comment_lines += 1
                comment_rows.append(line)
                continue
            # split the line by "\t"
            values = line.split("\t")
            if line_idx == 0:
                # the first line is the column names
                column_names = values
            else:
                if len(values) < len(column_names):
                    # fill the missing values with ""
                    values = values + [""] * (len(column_names) - len(values))
                elif len(values) > len(column_names):
                    # fill the missing values with ""
                    values = values[:len(column_names)]
                else:
                    pass
                all_values.append(values)
            line_idx += 1

    all_values_df = pd.DataFrame(all_values)
    if len(all_values_df) == 0:
        return [], pd.DataFrame()
    all_values_df.columns = column_names

    # consider the table "data_cna.txt", which has column names are patient id or sample id
    basename = os.path.basename(data_file)
    if basename == "data_cna.txt" or basename == "data_log2_cna.txt":
        # only keep the first 10 columns
        # and the last column is the patient id or sample id
        # we do not need to list all columns since they are repetitive
        # all_values_df = all_values_df.iloc[:, :10]
        # column_names = column_names[:10]
        pass

    # ZW: clean column names causing name mismatch when loading the data
    # clean the columns so they do not have any space, midline, comma, etc, replace with "_"
    # all_values_df.columns = _column_names_clean(all_values_df.columns)

    # infer the data types of each column and build the statistics correspondingly
    columns = all_values_df.columns
    column_metadata_list = []
    for column in columns:
        try:
            data_type = _infer_data_type(all_values_df, column)
            missing_rate, n_unique, statistics = _build_statistics(all_values_df, column, data_type)
            column_metadata = {
                "name": column,
                "data_type": data_type,
                "statistics": statistics,
                "n_unique": n_unique,
                "missing_rate": missing_rate
            }
            column_metadata_list.append(column_metadata)
        except Exception as e:
            import traceback
            print(f"[ERROR] failed to process {column}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            pdb.set_trace()
    return column_metadata_list, all_values_df, comment_lines, comment_rows


def phase3():
    """
    Get the dataset metadata for all the datasets in cBioPortal.
    """
    # read the dataset
    dataset_metadata_dir = os.path.join("/shared/eng/zifengw2/DSWizard/benchmark_datasets/cBioPortal", "dataset_metadata")
    if not os.path.exists(dataset_metadata_dir):
        os.makedirs(dataset_metadata_dir)

    # load the study metadata
    study_metadata = pd.read_csv(os.path.join(output_dir, "study_metadata.csv"))
    study_metadata = study_metadata.drop_duplicates(subset=["dataset_id"]).reset_index(drop=True)
    print("number of datasets in total to be processed: ", len(study_metadata))
    
    for idx, row in tqdm(study_metadata.iterrows(), desc="Processing datasets", total=len(study_metadata)):
        try:
            type_of_cancer = row["type_of_cancer"]
            dataset_id = row["dataset_id"]
            dataset_description = row["description"]
            dataset = os.path.join(cbioportal_dir, dataset_id)
            dataset_metadata = {}
            dataset_metadata["dataset_id"] = dataset_id
            dataset_metadata["type_of_cancer"] = type_of_cancer
            dataset_metadata["description"] = dataset_description
            dataset_metadata["tables"] = []
            # get all the text files under the dataset directory starting with "data_"
            data_files = glob(os.path.join(dataset, "data_*.txt"))
            for data_file in tqdm(data_files, desc="Processing tables for dataset %s" % dataset_id):
                table_metadata = {}
                table_metadata["name"] = os.path.basename(data_file)
                table_metadata["description"] = ""
                # read the text file
                # skip the line starting with "#"
                columns, all_values_df, n_comment_rows, comment_rows = _build_columns_metadata(data_file)
                if len(columns) == 0 or len(all_values_df) == 0:
                    # skip the table if it is empty
                    continue
                table_metadata["n_comment_rows"] = n_comment_rows # number of comment rows
                table_metadata["comment_rows"] = comment_rows # comment rows
                table_metadata["columns"] = columns # column names
                table_metadata["n_rows"] = len(all_values_df) # number of rows
                table_metadata["n_columns"] = len(all_values_df.columns) # number of columns
                dataset_metadata["tables"].append(table_metadata)
            output_path = os.path.join(dataset_metadata_dir, f"{dataset_id}.json")
            with open(output_path, "w") as f:
                json.dump(dataset_metadata, f, indent=4)
        except Exception as e:
            print(f"[ERROR] failed to process {dataset_id}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            pass
        """
        {{
            "id": the id of the dataset
            "name": the name of the dataset
            "tables": a list of tables in the dataset
                [
                    {{
                        "name": the name of the table,
                        "n_rows": the number of rows in the table
                        "n_columns": the number of columns in the table
                        "columns": a list of columns in the table
                        [
                            {{
                                "name": the name of the column
                                "data_type": the data type of the column, inferred from the column values
                                "statistics": the statistics of the column, for categorical/text/other columns, it is the number of unique values, and the top 10 frequent values, for numerical columns, it is the quantiles, for boolean columns, it is the number of True and False values.
                                "missing_rate": the percentage of missing values in the column
                            }},
                            ...
                        ]
                    }}
                    ...
                ]
        }}
        
        """
    pass


def phase4():
    """
    Get the dataset statistics for all the datasets in cBioPortal.
    how many tables, columns, rows, etc.
    """
    # load the dataset metadata
    dataset_metadata_dir = os.path.join("/shared/eng/zifengw2/DSWizard/benchmark_datasets/cBioPortal", "dataset_metadata")
    dataset_metadata_files = glob(os.path.join(dataset_metadata_dir, "*.json"))
    all_dataset_metadata = []
    for dataset_metadata_file in dataset_metadata_files:
        with open(dataset_metadata_file, "r") as f:
            dataset_metadata = json.load(f)
            dataset_id = dataset_metadata["dataset_id"]
            type_of_cancer = dataset_metadata["type_of_cancer"]
            num_tables = len(dataset_metadata["tables"])
            total_rows = 0
            total_columns = 0
            for table in dataset_metadata["tables"]:
                num_rows = table["n_rows"]
                num_columns = table["n_columns"]
                if num_columns > 1000:
                    # mostly tables data_cna.txt, or data_log2_cna.txt have more than 1000 columns
                    print(f"dataset {dataset_id} has {num_columns} columns in table {table['name']}")
                    # pdb.set_trace()
                    # pass
                total_rows += num_rows
                total_columns += num_columns
            metadata = {
                "dataset_id": dataset_id,
                "type_of_cancer": type_of_cancer,
                "num_tables": num_tables,
                "total_rows": total_rows,
                "total_columns": total_columns
            }
            all_dataset_metadata.append(metadata)
    all_dataset_metadata = pd.DataFrame(all_dataset_metadata)
    all_dataset_metadata.to_csv(os.path.join(output_dir, "dataset_statistics.csv"), index=False)

    # Group by cancer type and calculate statistics
    cancer_type_stats = all_dataset_metadata.groupby('type_of_cancer').agg({
        'dataset_id': 'count',  # Count number of datasets
        'num_tables': ['sum', 'mean'],  # Sum and average of tables
        'total_rows': 'sum',
        'total_columns': 'sum'
    }).round(2)

    # Rename columns for clarity
    cancer_type_stats.columns = [
        'total_datasets',
        'total_tables',
        'avg_n_table_per_dataset',
        'total_rows',
        'total_columns'
    ]

    # Sort by total_datasets in descending order
    cancer_type_stats = cancer_type_stats.sort_values('total_datasets', ascending=False)
    
    # Save cancer type statistics
    cancer_type_stats.to_csv(os.path.join(output_dir, "cancer_type_statistics.csv"))


def phase5():
    # map pmid to pmc full text if it is available
    # load the pubmed data
    pubmed_data = pd.read_csv(os.path.join(output_dir, "pubmed_data.csv"))
    pmids = pubmed_data["PMID"].unique().tolist()
    results = _fetch_pmc_data(pmids)

    pmc_texts = []
    for pmid, data in results:
        if data is None:
            continue
        # parse the data split by the section
        passages = data[0]['documents'][0]['passages']
        filtered_passages = []
        for passage in passages:
            sec_title = passage['infons']['section_type']
            if sec_title in ["RESULTS"]:
                filtered_passages.append(passage['text'])
        # join the filtered passages
        pmc_text = " ".join(filtered_passages)
        pmc_texts.append({"PMID": pmid, "Results": pmc_text})
    pmc_texts = pd.DataFrame(pmc_texts)
    pubmed_data = pubmed_data.merge(pmc_texts, on="PMID", how="left")
    pubmed_data.to_csv(os.path.join(output_dir, "pubmed_data.csv"), index=False)

    study_metadata = pd.read_csv(os.path.join(output_dir, "study_metadata.csv"))
    study_metadata = study_metadata.merge(pubmed_data[["PMID", "Results"]], on="PMID", how="left")
    study_metadata.to_csv(os.path.join(output_dir, "study_metadata.csv"), index=False)

if __name__ == "__main__":
    phase1()
    phase2()
    phase3()
    phase4()
    phase5()