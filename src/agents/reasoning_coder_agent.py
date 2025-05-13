"""
DataVoyager (O3-mini) plus one step code generation.
"""
import pdb
import re
from typing import List, Any, Dict
from langgraph.graph import Graph, StateGraph, END, START
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.messages import SystemMessage, HumanMessage

from tools.DockerSandbox import DockerSandboxTool
from .BaseAgent import BaseAgent, run_with_retry
from .state import (
    FinalResponse,
    CodeResult,
    AgentState,
    UserRequest,
    AnalysisPlan,
    FinalResponseForStructuring 
)
from .BaseAgent import cut_off_tokens
from agents import FINAL_ANSWER_PROMPT_TEMPLATE

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ANALYSIS_PLAN_PROMPT_TEMPLATE = """
# TASK  
Generate an analysis plan to evaluate the user's scientific hypothesis using the datasets provided.

The plan should consist of clear, actionable steps that can be **easily converted to {language} code** without needing any additional information.

# REQUIREMENTS  
- Use only table and column names from the schema—do not invent or guess names.  
- Ensure every step is unambiguous and directly executable.  
- Use consistent naming for all variables (e.g., tables, columns) throughout the plan.  
- Be as concise as possible while maintaining full clarity and precision.

# DATASET PATHS  
{dataset_paths}

# DATASET SCHEMA  
{dataset_schema}

# OUTPUT FORMAT  
Wrap the analysis plan in <analysis_plan> </analysis_plan> tags. So an example output would be
```
<analysis_plan>
1. load the dataset
2. print hello world
</analysis_plan>
```
"""
ANALYSIS_PLAN_PROMPT_TEMPLATE = PromptTemplate(
    template=ANALYSIS_PLAN_PROMPT_TEMPLATE,
    input_variables=["dataset_paths", "dataset_schema", "language"]
)

CODE_GENERATION_PROMPT_TEMPLATE = """
# TASK
Given the user-provided analysis plan for the user's scientific hypothesis, you **Must** write {language} code to fulfill the plan so that user can execute the code later
to evaluate the hypothesis.

# IMPORTANT: CODE OUTPUT REQUIREMENTS
You must import all the necessary libraries at the beginning of your code.

You must use explicit print() statements for ALL outputs you want to see or analyze. Simply writing expressions like 'df.head()' will NOT show results in the execution log. Always use:
- print(df.head())
- print(analysis_result)
- print(statistical_test_output)
Every intermediate result and final output must be wrapped in a print() statement to be visible in the execution log.


# DATASET PATHS
{dataset_paths}

## Ouptut
Your output should be in Markdown format and you should wrap the generated code in ```{language} ``` tags.
"""
CODE_GENERATION_PROMPT_TEMPLATE = PromptTemplate(
    template=CODE_GENERATION_PROMPT_TEMPLATE,
    input_variables=["dataset_paths", "language"]
)


class ReasoningCoderAgent(BaseAgent):
    
    name = "reasoning_coder_agent"

    def __init__(
        self, 
        api_type: str,
        api_key: str,
        endpoint: str,
        default_tools: bool = False,
        language: str = "python",
        planning_model: str = "gpt-4o",
        coding_model: str = "gpt-4o",
        final_response_model: str = "gpt-4o",
        **kwargs
    ):
        super().__init__(
            api_type=api_type,
            api_key=api_key,
            endpoint=endpoint,
            **kwargs
        )
        
        self.planning_model = planning_model
        self.coding_model = coding_model
        self.final_response_model = final_response_model

        # allows langgraph to use defaults
        self.system_prompt = None
        
        # TODO: add support for R/SAS
        assert language in ["python"], "Language is not supported"
        self.language = language

        # TODO add support for Docker Env Execution tool (Python/R)
        # TODO add support for SAS Execution tool
        self.default_tools = default_tools
        if (default_tools):
            self.tools = [
                PythonREPLTool()
            ]
        else:
            self.tools = []
    
    def withTools(self, tools: List[BaseTool]):
        """
        A function to set the tools for the agent.
        
        Args:
            tools: The tools to set for the agent
        """
        self.tools = tools
        return self

    # def _validate_chat_history(self, messages: Sequence[BaseMessage]):
    #     pass

    # def call_model(self, state: AgentState):
    #     self._validate_chat_history(state.messages)
    #     response = cast(AIMessage, self.model_runnable.invoke(state))

    #     # add agent name to the AIMessage
    #     response.name = self.name

    #     # We return a list, because this will get added to the existing list
    #     return {"messages": [response]}
    
    def withDefaultSystemPrompt(
        self,
        dataset_paths: str,
        dataset_schema: str
    ):
        self.analysis_plan_prompt = ANALYSIS_PLAN_PROMPT_TEMPLATE.format(
            dataset_paths=dataset_paths,
            dataset_schema=dataset_schema,
            language=self.language
        )
        
        self.code_generation_prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
            dataset_paths=dataset_paths,
            language=self.language
        )
        
        # placeholder
        self.final_answer_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            observations="TBD"
        )
        
        return self

    def generate_analysis_plan(
        self,
        state: UserRequest,
        **kwargs
        ) -> AgentState:
        """
        A function to generate the data analysis code.
        """
        # input
        input_query = state["input_query"]
        
        messages = [
            SystemMessage(content=self.analysis_plan_prompt),
            HumanMessage(content=input_query)
        ]

        llm = self.get_model(
            api=self.api_type,
            model_name=self.planning_model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            reasoning_effort="low",
            **kwargs
        )

        # generate analysis plan
        result = run_with_retry(llm.invoke, arg=messages)
        plan = result.content
        
        return AnalysisPlan(
            input_query=input_query,
            plan=plan
        )

    def generate_code(
        self,
        state: AnalysisPlan,
        **kwargs
    ) -> CodeResult:
        """
        A function to generate the data analysis code.
        """
        if isinstance(state, dict):
            state = AnalysisPlan(**state)

        plan = state.plan
        
        input_query = state.input_query

        messages = [
            SystemMessage(content=self.code_generation_prompt),
            HumanMessage(content=f"Hypothesis: {input_query}\n\nAnalysis Plan: {plan}")
        ]

        # use gpt-4o for code generation
        llm = self.get_model(
            api=self.api_type,
            model_name=self.coding_model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            **kwargs
        )

        # generate code
        result = run_with_retry(llm.invoke, arg=messages)
        
        # extract code from ```python tags if there are multiple ```python tags, combine them
        code = result.content
        code_blocks = re.findall(rf"```{self.language}(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
        combined_code = "\n\n".join(block.strip() for block in code_blocks)

        # execute the code
        docker_tool: DockerSandboxTool = self.tools[0]

        execution_result = docker_tool._run(
            language=self.language,
            code=combined_code
        )

        # cut off the execution results to 10000 tokens
        # in case it is too long and makes huge cost
        stdout = cut_off_tokens(execution_result["output"], 10000)

        return CodeResult(
            input_query=input_query,
            plan=plan,
            code=combined_code,
            execution_results=stdout
        )
    
    def generate_structured_response(
        self,
        state: CodeResult,
        **kwargs
        ) -> FinalResponse:
        """
        A function to generate the structured response.
        """
        # use gpt-4o for final answer generation
        llm = self.get_model(
            api=self.api_type,
            model_name=self.final_response_model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_completion_tokens=5000,
            **kwargs
        )
        model_with_structured_output = llm.with_structured_output(FinalResponseForStructuring)

        observations = state.execution_results
        hypothesis = state.input_query
        analysis_plan = state.plan
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
            analysis_plan=analysis_plan,
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
        """
        A function to create the agent graph.
        """
        workflow = StateGraph(
            AgentState,
            input=UserRequest,
            output=FinalResponse
        )
        # add the node
        workflow.add_node("generate_analysis_plan", self.generate_analysis_plan)
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("generate_structured_response", self.generate_structured_response)

        # add the edges
        workflow.add_edge(START, "generate_analysis_plan")
        workflow.add_edge("generate_analysis_plan", "generate_code")
        workflow.add_edge("generate_code", "generate_structured_response")
        workflow.add_edge("generate_structured_response", END)

        # compile the graph
        self.agent_graph = workflow.compile(debug=debug, name=self.name)
        return self