import time
from langchain.tools import BaseTool
from tools.DockerSandbox.sandbox_core.ExecutionSandboxWrapper import (
    DEFAULT_REMOTE_PATH,
    SANDBOX_IMANGE_IDENTIFIER,
    ExecutionSandboxWrapper,
    EvalDataset
)
from typing import Literal, Union, Type, Any, Optional
from pydantic import BaseModel, Field
import logging
from agents.BaseAgent import cut_off_tokens

# Configure logger to output to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class DockerSandboxToolArgs(BaseModel):
    language: Union[Literal["r"], Literal["python"]]
    """The programming language to execute the code in. Must be either 'R' or 'Python'."""
    code: str 
    """The source code to execute in the Docker container."""

class DockerSandboxTool(BaseTool):
    """
    A tool that allows you to execute code in a Docker container.
    
    Supports Python and R.
    """
    
    args_schema: Type[DockerSandboxToolArgs] = DockerSandboxToolArgs
    name: str = "docker_sandbox"
    description: str = "Use this tool to execute code in a Docker container"
    response_format: Literal["content", "content_and_artifact"] = "content"
    
    # Add field declarations for instance attributes
    image_identifier: str = Field(default=SANDBOX_IMANGE_IDENTIFIER)
    target_dir: str = Field(default="")
    dataset: Optional[EvalDataset] = Field(default=None)
    verbose: bool = Field(default=False)
    sandbox: Optional[ExecutionSandboxWrapper] = Field(default=None)
    sandbox_container_id: Optional[str] = Field(default=None)
    metrics: dict = Field(default_factory=lambda: {
        "total_execution_time": 0,
        "total_evaluations": 0,
        "history": [],
        "available_files": []
    })
    clip_output: int = Field(default=-1)
    
    def __init__(
        self,
        image_identifier: str = SANDBOX_IMANGE_IDENTIFIER,
        verbose: bool = False,
        workdir: str = DEFAULT_REMOTE_PATH,
        clip_output: int = -1
    ):
        super().__init__()
        self.image_identifier = image_identifier
        self.verbose = verbose # override the BaseTool verbose
        self.clip_output = clip_output
        
        # initialize the sandbox
        self.sandbox = ExecutionSandboxWrapper(image_identifier=self.image_identifier, workdir=workdir)
        self.sandbox_container_id = self.sandbox.container.id
        logger.debug(f"SUCCESS: Execution sandbox created - id: `{self.sandbox_container_id}`")
        
        # initialize metrics
        self.metrics = {
            "total_execution_time": 0,
            "total_evaluations": 0,
            "history": [],
            "available_files": []
        }
        
    def upload_tables(self, dataset: EvalDataset):
        """
        insert the tables into the sandbox
        """
        
        # check if the dataset is already uploaded
        if all(table_path in self.sandbox.available_files for table_path in dataset.target_table_paths):
            logger.debug(f"SUCCESS: Data already uploaded to sandbox.")
            return
        
        success = self.sandbox.upload_tables(dataset=dataset)
        assert success, "Failed to upload tables to sandbox"
        logger.debug(f"SUCCESS: Data successfully placed in sandbox.")
        self.metrics["available_files"].append(self.sandbox.available_files)
    
    def __del__(self):
        try:
            self.sandbox.stop()
        except Exception as e:
            logger.exception(f"Failed to stop sandbox: {e}")
        
    def _run(self, language: str, code: str):
        
        self.metrics["total_evaluations"] += 1
        start = time.time()
        exit_code, output, artifacts = self.sandbox.execute(
            language=language,
            code=code
        )
        
        output = output.decode('utf-8')
        # useful for llm tools
        if (self.clip_output > -1):
            output = cut_off_tokens(output, self.clip_output)
        
        end = time.time()
        total_execution_time = end - start
        self.metrics["total_execution_time"] += total_execution_time
        self.metrics["history"].append({
            "language": language,
            "code": code,
            "exit_code": exit_code,
            "output": output,
            "artifacts": artifacts,
            "execution_time": total_execution_time
        })
        
        return {
            "exit_code": exit_code,
            "output": output,
            "artifacts": artifacts,
            "execution_time": total_execution_time,
            # pass through the code and language
            "code": code,
            "language": language
        }


    def _arun(self, language: str, code: str):
        return self._run(language, code)
    
    
    
    