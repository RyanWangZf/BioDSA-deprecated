import shutil
import io
import os
import tarfile
import traceback
from typing import List, Tuple, Dict, Callable
import uuid
import docker
from docker.errors import NotFound
import threading
from datetime import datetime
from typing import Union
from pydantic import BaseModel
import pandas as pd
import logging

# Configure logger to output to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# executino sandbox constants
SANDBOX_IMANGE_IDENTIFIER = "combined-sandbox:latest" # this docker image must be built in advance
DEFAULT_REMOTE_PATH = "/workdir" # location inside docker container where data files (csvs) are kept

class Artifact(BaseModel):
    """
    Define the output artifact of code generation and execution
    """
    content: Union[bytes,str] = None # content of the artifact in bytes (like img) or string (like txt, html)
    file_name: str = None # the name of the artifact
    file_path: str = None # the path of the artifact in the local file system
    file_type: str = None # type of the artifact, e.g., "image", "csv", "json", "html", "pdf"

    def __str__(self) -> str:
        return f"""Artifact <{self.file_name}>"""

class EvalDataset:
    
    tables: Dict[str, pd.DataFrame] = {}
    
    def __init__(
        self, 
        local_table_paths: List[str], 
        target_table_paths: List[str],
        table_preprocessing_fn: Callable = None
    ):
        """
        This class is used to create a dataset for the execution sandbox & agents.
        The local tables are loaded into memory as pd data frames, so they can be written into the sandbox at the target locations.
        
        Args:
            local_table_paths: the paths to the tables on the local machine
            target_table_paths: the paths to the target tables in the sandbox
            table_preprocessing_fn: a function to preprocess the tables. 
                This function should take a pd.DataFrame and return a pd.DataFrame.
                This is useful for cleaning the column names of the tables.
            
        Note: paths should have the revelant extension for the file type, e.g. ".csv", ".txt", ".tsv"
        """
        
        assert len(local_table_paths) == len(target_table_paths), "local_table_paths and target_table_paths must have the same length"
        
        self.local_table_paths = local_table_paths
        self.target_table_paths = target_table_paths
        # self.table_preprocessing_fn = table_preprocessing_fn
        # self.local_tables = []
        # for path in local_table_paths:
        #     df = self.load_table(path)
        #     if self.table_preprocessing_fn is not None:
        #         df = self.table_preprocessing_fn(df)
        #     self.local_tables.append(df)
        
    # def load_table(self, table_path: str) -> pd.DataFrame:
    #     extension = os.path.splitext(table_path)[1]
    #     if extension == ".csv":
    #         return pd.read_csv(table_path)
    #     elif extension == ".txt":
    #         # Check if this is a cBioPortal clinical data file (has metadata comments)
    #         with open(table_path, 'r') as f:
    #             first_line = f.readline().strip()
    #             if first_line.startswith('#'):
    #                 # Skip all comment lines and read the actual data
    #                 return pd.read_csv(table_path, sep="\t", comment='#', low_memory=False)
    #         return pd.read_csv(table_path, sep="\t", low_memory=False)
    #     elif extension == ".tsv":
    #         return pd.read_csv(table_path, sep="\t", low_memory=False)
    #     else:
    #         raise ValueError(f"Unsupported file extension: {extension}")
        
    def __len__(self):
        return len(self.local_tables)
    
    def __getitem__(self, index: int):
        return self.local_tables[index], self.target_table_paths[index]
    
    def __iter__(self):
        return iter(zip(self.local_table_paths, self.target_table_paths))
    
    def __str__(self):
        return f"EvalDataset with {len(self)} tables"
    
    def __repr__(self):
        return self.__str__()
    
    


class ExecutionSandboxWrapper:

    def __init__(self, image_identifier: str, workdir: str):
        """
        Start a container with the specified image

        `target_dir` is the workspace for all execution sandbox activities. 
        This variable needs to be set on initialization, since all the sandbox functions require this path.
        If this path changes at some point while using this API, then operations like `start` and `execute` will fail.
        """
        self.workdir = workdir
        
        client = docker.from_env()
        try:
            container = None
            try:
                container = client.containers.run(image_identifier, detach=True, network_disabled=False)
            except Exception as e:
                print(f"Error starting container: {e}")
                print(f"Container: {container}")
                print(traceback.format_exc())
                raise e

            if (container is not None):
                self.image = container.image
                self.container = container
            else: 
                raise Exception("Container not started")
            
            self.available_files = []
            self.all_artifact_files = []
        finally:
            client.close()
        
        
    def upload_tables(self, dataset: EvalDataset) -> bool:
        """
        place the tables in the dataset into the docker container
        """

        if self.container is None:
            raise Exception("Container not started")

        # write each dataframe to the docker container
        for local_table_path, target_path in dataset:
            try:
                # Use the current thread ID and timestamp to create a unique identifier
                unique_id = f"{threading.get_ident()}_{int(datetime.now().timestamp() * 1000)}"

                # step 1: create a temp file on the local machine
                temp_path = f'/tmp/table_{unique_id}'
                
                # copy the local table object to the temp file
                shutil.copy(local_table_path, temp_path)

                # step 2: create a local tar file with the temp file
                file_directory = os.path.dirname(target_path)
                
                # initialize the container workspace if it doesn't exist
                self.container.exec_run(f'mkdir -p {file_directory}')
                
                file_name = os.path.basename(target_path)
                
                tar_path = f'/tmp/table_{unique_id}.tar'
                with tarfile.open(tar_path, 'w') as tar:
                    tar.add(temp_path, arcname=file_name)

                # step 3: copy the tar file to the container **at the target directory**
                with open(tar_path, 'rb') as f:
                    data = f.read()
                    self.container.put_archive(file_directory, data)

                os.remove(tar_path)
                os.remove(temp_path)

                # Clean up the temporary files after use
                self.available_files.append(target_path)
            except Exception as e:
                logger.error(f"Error uploading table {local_table_path} to {target_path}: {e}")
                raise e

        return True

    def execute(self, language: str, code: str) -> Tuple[int, str, List[str]]:
        """
        get the dataframes and extract any resulting files/figures/stdout

        Returns the exit code, stdout, and a list of artifact paths ON HOST MACHINE.

        Artifacts are in the `/tmp` directory of the host machine
        """
        # generate a filename for the code in the container
        execution_id = uuid.uuid4().hex[:8]

        # copy the code into the container as file
        host_file_path = f'/tmp/{execution_id}_dswiz'
        if (language == "python"):
            host_file_path += '.py'
        elif (language == "r"):
            host_file_path += '.r'

        host_tar_file = f'/tmp/{execution_id}.tar'

        with open(host_file_path, 'w') as f:
            f.write(code)

        arcname = f'{execution_id}'
        if (language == "python"):
            arcname += '.py'
        elif (language == "r"):
            arcname += '.r'

        with tarfile.open(host_tar_file, 'w') as tar:
            tar.add(host_file_path, arcname=arcname)

        self.container.exec_run('mkdir /code')
        with open(host_tar_file, 'rb') as f:
            self.container.put_archive('/code', f)

        os.remove(host_file_path)
        os.remove(host_tar_file)

        if language == "python":
            exit_code, output = self.container.exec_run(
                f'python /code/{execution_id}.py', workdir=self.workdir)
        elif language == "r":
            exit_code, output = self.container.exec_run(
                f'Rscript /code/{execution_id}.r', workdir=self.workdir)

        new_files = self.container.exec_run(
            f'ls {self.workdir}').output.decode('utf-8').split('\n')

        new_files_set = set()
        for file in new_files:
            if file != '' and (".csv" not in file and ".tsv" not in file and ".txt" not in file):
                new_files_set.add(file)


        # the mechanism for surfacing any artifacts resulting from the execution. 
        # create a new folder in /tmp with the execution_id
        artifacts = []
        host_folder = os.path.join('/tmp', execution_id)
        os.makedirs(host_folder, exist_ok=True)
        
        self.all_artifact_files.append(host_folder) # we need to track this folder so we can remove it later. Otherwise there will be a memory leak.
        
        for file in new_files_set:
            # get the object out of the docker container, and load it to the host file system.
            # the file name should be the same as the one in the container
            host_file_path = os.path.join(host_folder, file)

            # copy from docker container to host file system
            bits, _ = self.container.get_archive(f'{self.workdir}/{file}')
            tar_stream = io.BytesIO(b''.join(bits))

            try:
                with tarfile.open(fileobj=tar_stream) as tar:
                    tar.extractall(path=os.path.dirname(host_file_path))
            finally:
                tar_stream.close()
            
            # track all the files in the host system.
            # when running multiple experiments, this will help us clean up the files.
            artifacts.append(host_file_path)
            self.all_artifact_files.append(host_file_path)
        
        return exit_code, output, artifacts

    def stop(self):
        """
        Stop the docker container and clean up resources
        """
        # try to remove all the files in the all_artifact_files list
        # if we do not remove these, the host machine will run out of disk space
        artifacts_removed = []
        for file in self.all_artifact_files:
            if not os.path.exists(file):
                artifacts_removed.append(file)
                continue
            
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    artifacts_removed.append(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    
                    if os.path.exists(file):
                        logger.warning(f"Directory still exists after rmtree: {file}")
                        # Try force remove with shell command as fallback
                        os.system(f"rm -rf {file}")
                        
                    artifacts_removed.append(file)
                else:
                    logger.warning(f"File {file} is not a file or directory")
            except Exception as e:
                logger.error(f"Error removing file {file}: {e}")
        
        logger.info(f"Removed {len(artifacts_removed)} artifacts of {len(self.all_artifact_files)}")
        self.all_artifact_files = [
            file for file in self.all_artifact_files if file not in artifacts_removed
        ]
        
        client = docker.from_env()
        try:
            try:
                # Stop and remove container
                container = client.containers.get(self.container.id)
                container.stop(timeout=120)
                container.remove()    
            except NotFound as e:
                logging.warning(f"Container not found: {e}")
            except Exception as e:
                logging.exception(f"Error stopping container: {e}")
                raise e
            
            # Prune unused volumes
            try:
                client.volumes.prune()
                logging.info("Successfully pruned unused volumes")
            except Exception as e:
                logging.warning(f"Failed to prune volumes: {e}")
                
        finally:
            client.close()
        


    def clear_code(self):
        """
        Clear only the code directory in the container
        """
        self.container.exec_run('rm -rf /code')

    def clear_workspace(self):
        """
        Clear the workspace of the docker container while preserving uploaded tables.
        Uses the available_files list to determine which files to preserve.
        """
        # First, clear the code directory
        self.clear_code()
        
        if self.available_files:
            # Create a find command that excludes the specific files we want to preserve
            exclude_patterns = ' '.join([f'! -path "{f}"' for f in self.available_files])
            self.container.exec_run(f'find {self.workdir} -type f {exclude_patterns} -delete')
        else:
            # If no files to preserve, clear everything
            self.container.exec_run(f'rm -rf {self.workdir}/*')
            
            
if __name__ == "__main__":
    sandbox = ExecutionSandboxWrapper(SANDBOX_IMANGE_IDENTIFIER, DEFAULT_REMOTE_PATH)
    print(sandbox.container.id)