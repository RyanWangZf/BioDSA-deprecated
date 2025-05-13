from typing import Dict, Any
from pydantic import BaseModel
from typing import List, Annotated, Sequence, TypeVar, cast
from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.utils.runnable import RunnableCallable
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from agents.state import FinalResponse, AgentState, CodeResult, UserRequest, FinalResponseForStructuring
from agents.BaseAgent import BaseAgent, run_with_retry
from agents.BaseAgent import cut_off_tokens
from agents import FINAL_ANSWER_PROMPT_TEMPLATE
import re
from tools.DockerSandbox.DockerSandboxTool import DockerSandboxTool

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SYSTEM_PROMPT_DS_WIZARD_CODER_AGENT_TEMPLATE = """
# TASK
Given the user-provided scientific hypothesis, you **Must** write {language} code to help the user evaluate the hypothesis.

# IMPORTANT: CODE OUTPUT REQUIREMENTS
You must import all the necessary libraries at the beginning of your code.

You must use explicit print() statements for ALL outputs you want to see or analyze. Simply writing expressions like 'df.head()' will NOT show results in the execution log. Always use:
- print(df.head())
- print(analysis_result)
- print(statistical_test_output)
Every intermediate result and final output must be wrapped in a print() statement to be visible in the execution log.


# DATASET PATHS
{dataset_paths}

# DATASET SCHEMA  
{dataset_schema}

## Ouptut
Your output should be in Markdown format and you should wrap the generated code in ```{language} ``` tags.
"""

SYSTEM_PROMPT_DS_WIZARD_REACT_AGENT = PromptTemplate(
    template=SYSTEM_PROMPT_DS_WIZARD_CODER_AGENT_TEMPLATE,
    input_variables=["dataset_paths", "dataset_schema", "language", "tool_name"]
)

class CoderAgentV2(BaseAgent):
    
    name = "baseline_coder_agent"

    def __init__(
        self, 
        model_name: str, 
        final_response_model: str,
        api_type: str,
        api_key: str,
        endpoint: str,
        default_tools: bool = False,
        language: str = "python"
    ):
        super().__init__(
            model_name=model_name,
            api_type=api_type,
            api_key=api_key,
            endpoint=endpoint,
        )
        self.final_response_model = final_response_model
        
        # allows langgraph to use defaults
        self.system_prompt = None
        
        # TODO: add support for R/SAS
        assert language in ["python"], "Language is not supported"
        self.language = language
        
        # TODO add support for SAS Execution tool
        self.default_tools = default_tools
        if (default_tools):
            self.tools = [
                PythonREPLTool()
            ]
        else:
            self.tools = []
            
    def withDefaultSystemPrompt(self, dataset_paths: str, dataset_schema: str):
        """
        A function to construct the default system prompt for the agent.
        
        Args:
            query: The query to evaluate
            dataset_paths: The paths to the datasets
        """
        tool_name = self.tools[0].name
        
        self.system_prompt = SYSTEM_PROMPT_DS_WIZARD_CODER_AGENT_TEMPLATE.format(
            dataset_paths=dataset_paths,
            tool_name=tool_name,
            language=self.language,
            dataset_schema=dataset_schema,
        )
        
        # print(f"\n======= SYSTEM PROMPT =======\n{self.system_prompt}\n============================\n")
        
        # placeholder
        self.final_answer_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            observations="TBD"
        )
        
        return self
    
    def withTools(self, tools: List[BaseTool]):
        """
        A function to set the tools for the agent.
        
        Args:
            tools: The tools to set for the agent
        """
        self.tools = tools
        return self 
    
    def generate_code(
        self,
        state: UserRequest,
        **kwargs
    ) -> CodeResult:
        """
        A function to generate the code for the agent.
        """
        input_query = state["input_query"]
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Hypothesis: {input_query}")
        ]
        
        llm = self.get_model(
            api=self.api_type,
            model_name=self.model_name,
            api_key=self.api_key,
            endpoint=self.endpoint,
            **kwargs
        )
        
        
        result = run_with_retry(llm.invoke, arg=messages)
        
        code = result.content
        code_blocks = re.findall(rf"```{self.language}(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
        combined_code = "\n\n".join(block.strip() for block in code_blocks)
        
        docker_tool: DockerSandboxTool = self.tools[0]

        execution_result = docker_tool._run(
            language=self.language,
            code=combined_code
        )
        
        stdout = cut_off_tokens(execution_result["output"], 10000)
        
        return CodeResult(
            input_query=input_query,
            code=combined_code,
            plan="", # unused here
            execution_results=stdout
        )
    
    def generate_structured_response(
        self,
        state: CodeResult,
        **kwargs
    ) -> FinalResponse:
        model_with_structured_output = self.llm.with_structured_output(FinalResponse)
        
        # if it is a tool message, we want to cut it off in case it is too long
        llm = self.get_model(
            api=self.api_type,
            model_name=self.model_name,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_completion_tokens=5000,
            **kwargs
        )
        
        model_with_structured_output = llm.with_structured_output(FinalResponseForStructuring)
        
        observations = state.execution_results
        hypothesis = state.input_query
        code = state.code
        
        self.final_answer_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            observations=observations
        )
        
        messages = [
            SystemMessage(content=self.final_answer_prompt),
            HumanMessage(content=f"Hypothesis: {hypothesis}")
        ]
        
        response = run_with_retry(model_with_structured_output.invoke, arg=messages)
        
        return FinalResponse(
            hypothesis=hypothesis,
            observations=observations,
            analysis_plan="", # unused here
            code=code,
            language=self.language,
            final_answer=response.final_answer,
            evidence=response.evidence
        )
        
    def generate(
        self,
        input_query: str
    ) -> Dict[str, Any]:
        """
        Override the base method for generating code.
        
        Args:
            input_query: The user query to process
            stream_mode: The mode to stream the code
            **kwargs: Additional arguments to pass to the agent graph
        """
        assert self.agent_graph is not None, "Agent graph is not set"
        
        # Extract input_query from kwargs
        if input_query is None:
            return {"error": "input_query is required"}
        
        try:
            inputs = {
                "input_query": input_query
            }
        
            # Invoke the agent graph and return the result
            result = self.agent_graph.invoke(
                inputs
            )
            return result
            
        except Exception as e:
            print(f"Error streaming code: {e}")
            raise e
    
    def create_agent_graph(self, debug: bool = False) -> Graph:
        
        if self.system_prompt is None:
            raise ValueError("System prompt is not set")
    
        prompt_runnable = RunnableCallable(
            lambda state: [self.system_prompt] + state.messages,
            name="Prompt",
        )
        
        # ~~~ Define the Agent ~~~
        # bind the tools to the model
        model = self.llm.bind_tools(self.tools)
        
        # the runnable that will be used for the agent
        self.model_runnable = prompt_runnable | model
    
        # the actual agent workflow graph
        workflow = StateGraph(
            AgentState,
            input=UserRequest,
            output=FinalResponse
        )
        
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("generate_structured_response", self.generate_structured_response)
        
        workflow.add_edge("generate_code", "generate_structured_response")
        workflow.add_edge("generate_structured_response", END)
        
        workflow.set_entry_point("generate_code")
        
        self.agent_graph = workflow.compile(
            debug=debug,
            name=self.name
        )
        
        return self