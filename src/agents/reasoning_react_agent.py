import json
from agents.BaseAgent import BaseAgent
from typing import List, Sequence, Union, cast, Dict, Any
from langgraph.graph import Graph, StateGraph, END
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from pydantic import BaseModel
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable, RunnableConfig

from agents.state import FinalResponse, FinalResponseForStructuring
from agents.BaseAgent import BaseAgent, run_with_retry
from agents import FINAL_ANSWER_PROMPT_TEMPLATE

AGENT_MODEL_PROMPT_TEMPLATE = """
You are a scientific agent who can plan and execute python code iteratively to evaluate a scientific hypothesis.

Note:
- You must execute and refine the given analysis plan iteratively until you have enough evidence to support the hypothesis.
- You must always write a single Python code block that can be executed directly based on the analysis plan.
- Use `print()` statements in your code to get the observations.
"""

PLANNING_PROMPT_TEMPLATE = """
# TASK  
Generate an analysis plan to evaluate the user's scientific hypothesis using the datasets provided.

The plan should consist of clear, actionable psudo codesteps that can be **easily converted to python code** without needing any additional information.

# REQUIREMENTS 
• Use only table and column names from the schema—do not invent or guess names.  
• Ensure every step is unambiguous and directly executable.  
• Use consistent naming for all variables (e.g., tables, columns) throughout the plan.  
• Be as concise as possible while maintaining full clarity and precision.

# DATASET PATHS
{dataset_paths}

# DATASET SCHEMA
{dataset_schema}
"""

AGENT_MODEL_PROMPT = AGENT_MODEL_PROMPT_TEMPLATE

PLANNING_PROMPT = PromptTemplate(
    template=PLANNING_PROMPT_TEMPLATE,
    input_variables=["query", "dataset_paths", "dataset_schema"]
)

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    structured_response: FinalResponse | None = None
    remaining_steps: int
    is_last_step: bool = False
    hypothesis: str
    analysis_plan: str

class ReasoningReactAgent(BaseAgent):
    name = "reasoning_react_agent"

    def __init__(
        self, 
        plan_model_name: str, 
        agent_model_name: str,
        final_response_model: str,
        api_type: str,
        api_key: str,
        endpoint: str,
        default_tools: bool = False,
        language: str = "python",
    ):
        super().__init__(
            model_name=plan_model_name,
            api_type=api_type,
            api_key=api_key,
            endpoint=endpoint,
        )
        
        self.plan_model_name = plan_model_name
        self.agent_model_name = agent_model_name
        
        self.agent_model = self.llm # use the model defined in the parent class
        self.plan_model = self.get_model(
            api=self.api_type,
            model_name=self.agent_model_name,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_completion_tokens=5000,
        )
        
        self.final_response_model = final_response_model
        
        assert language in ["python"], "Language is not supported"
        self.language = language
        
        self.default_tools = default_tools
        if (default_tools):
            self.tools = [PythonREPLTool()]
        else:
            self.tools = []
                        
    def withDefaultSystemPrompt(self, query: str, dataset_paths: str, dataset_schema: str):
        """
        A function to construct the default system prompt for the agent.
        
        Args:
            query: The query to evaluate
            dataset_paths: The paths to the datasets
            dataset_schema: The schema of the datasets
        """
        
        self.plan_model_prompt = PLANNING_PROMPT.format(
            query=query,
            dataset_paths=dataset_paths,
            dataset_schema=dataset_schema
        )
        
        self.agent_model_prompt = AGENT_MODEL_PROMPT_TEMPLATE
        
        # placeholder
        self.final_answer_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            observations="TBD"
        )
        
        return self
    
    def withTools(self, tools: List[BaseTool]):
        self.tools = tools
        return self
    
    def _validate_chat_history(self, messages: Sequence[BaseMessage]):
        """Validate that all tool calls in AIMessages have a corresponding ToolMessage."""
        all_tool_calls = [
            tool_call
            for message in messages
            if isinstance(message, AIMessage)
            for tool_call in message.tool_calls
        ]
        tool_call_ids_with_results = {
            message.tool_call_id for message in messages if isinstance(message, ToolMessage)
        }
        tool_calls_without_results = [
            tool_call
            for tool_call in all_tool_calls
            if tool_call["id"] not in tool_call_ids_with_results
        ]
        if tool_calls_without_results:
            raise ValueError(
                "Found AIMessages with tool_calls that do not have a corresponding ToolMessage. "
                f"Here are the first few of those tool calls: {tool_calls_without_results[:3]}.\n\n"
                "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage "
                "(result of a tool invocation to return to the LLM) - this is required by most LLM providers."
            )
    
    def generate_analysis_plan(self, state: AgentState) -> AgentState:
        
        response = cast(AIMessage, run_with_retry(self.plan_model_runnable.invoke, arg=state))
        response.name = "analysis_plan"
        e
        # Manually create a list of messages starting with the system prompt
        current_messages = [SystemMessage(content=self.plan_model_prompt)] + state.messages
        
        return AgentState(
            messages=current_messages + [response], # this is the first message in the chat history
            remaining_steps=state.remaining_steps,
            structured_response=state.structured_response,
            hypothesis=state.hypothesis,
            analysis_plan=response.content
        )
        
    def call_agent_model(self, state: AgentState) -> AgentState:
        response = cast(AIMessage, run_with_retry(self.agent_model_runnable.invoke, arg=state))
        response.name = "agent_model"
        
        # Check if we need more steps
        if state.remaining_steps <= 1:
            return AgentState(
                messages=[AIMessage(content="Sorry, need more steps to process this request. Try to answer the question with the information you have so far.")],
                remaining_steps=0,
                structured_response=state.structured_response,
                is_last_step=True,
                hypothesis=state.hypothesis,
                analysis_plan=state.analysis_plan
            )
        
        return AgentState(
            messages=state.messages + [response],
            remaining_steps=state.remaining_steps - 1,
            structured_response=state.structured_response,
            is_last_step=state.is_last_step,
            hypothesis=state.hypothesis,
            analysis_plan=state.analysis_plan
        )
    
    def generate_structured_response(
        self,
        state: AgentState,
        **kwargs
    ) -> FinalResponse:
        model_with_structured_output = self.llm.with_structured_output(FinalResponse)
        
        # if it is a tool message, we want to cut it off in case it is too long
        llm = self.get_model(
            api=self.api_type,
            model_name=self.final_response_model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_completion_tokens=5000,
            **kwargs
        )
        
        model_with_structured_output = llm.with_structured_output(FinalResponseForStructuring)

        # construct the observations string.
        # we look through the messages and find the ToolMessage with the name "execute_code"
        # and then we take the content of the message
        observations = []
        code_strings = []
        for message in state.messages:
            if isinstance(message, ToolMessage):
                tool_message_content: Dict = json.loads(message.content) if message.content is not None else {}
                stdout = tool_message_content["output"]
                language = tool_message_content["language"]
                code_string = tool_message_content["code"]
                code_strings.append("<code>\n" + code_string + "\n</code>")
                observations.append(f"## Observation {len(observations)}\n### Code: \n```{language}\n{code_string}\n```\n### Stdout:\n{stdout}")
        
        # first message is the system prompt; second is the LLM generated analysis plan
        analysis_plan = state.messages[2].content
        
        observations = "\n".join(observations)
        code = "\n".join(code_strings)
        
        self.final_answer_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            observations=observations
        )
        
        messages = [
            SystemMessage(content=self.final_answer_prompt),
            HumanMessage(content=f"Hypothesis: {state.hypothesis}")
        ]
        
        response = run_with_retry(model_with_structured_output.invoke, arg=messages)
        
        return FinalResponse(
            hypothesis=state.hypothesis,
            observations=observations,
            analysis_plan=analysis_plan,
            code=code,
            language=self.language,
            final_answer=response.final_answer,
            evidence=response.evidence
        )
    
    def should_continue(self, state: AgentState) -> Union[str, list]:
        """
        Determine whether to continue the agent loop or end.
        
        This function is called after the model has responded to route the agent to the next step.
        """
        last_message = state.messages[-1]
        
        is_last_step = state.is_last_step
        
        # If no tool calls, we're done
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "generate_structured_response"
        
        # if last_message.tool_calls:
        #     print(f"Tool calls: {last_message.tool_calls}")
        # If we have no steps left, we're done
        if is_last_step:
            return "generate_structured_response"
            
        # Otherwise continue to tools
        return "tools"
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Overloaded method for generating code.
        
        Args:
            input_query: The user query to process
            **kwargs: Additional arguments to pass to the agent graph
            
        Returns:
            Dict[str, Any]: The result from the agent graph or an error dict
        """
        
        assert self.agent_graph is not None, "Agent graph is not set"
        
        # Extract input_query from kwargs
        input_query = kwargs.pop("input_query", None)
        if input_query is None:
            return {"error": "input_query is required"}
        
        remaining_steps = kwargs.pop("remaining_steps", 25)
        remaining_steps += 2 # for the human message and the analysis plan
        
        try:
            # Prepare inputs for agent graph
            inputs = {
                "messages": [
                    ("user", "Hypothesis: " + input_query)
                ],
                "hypothesis": input_query, # keep the raw hypothesis for the final answer
                "analysis_plan": "",
                "remaining_steps": remaining_steps,
                **kwargs,
            }
            
            # Invoke the agent graph and return the result
            result = self.agent_graph.invoke(inputs, RunnableConfig(recursion_limit=40))
            return result
            
        except Exception as e:
            print(f"Error generating code: {e}")
            raise e
    
    def create_agent_graph(self, debug: bool = False) -> Graph:
        if len(self.tools) == 0:
            raise ValueError("No tools provided")
        
        if self.agent_model_prompt is None:
            raise ValueError("Agent model prompt is not set")
        
        if self.plan_model_prompt is None:
            raise ValueError("Plan model prompt is not set")

        # PlanAgent: Create prompt runnable
        plan_prompt_runnable = RunnableCallable(
            lambda state: [self.plan_model_prompt] + state.messages,
            name="Plan Prompt",
        )
        
        self.plan_model_runnable = plan_prompt_runnable | self.plan_model

        # ReactAgent: Create prompt runnable
        agent_prompt_runnable = RunnableCallable(
            lambda state: [self.agent_model_prompt] + state.messages,
            name="Agent Prompt",
        )
        
        agent_model = self.agent_model.bind_tools(self.tools)
        self.agent_model_runnable = agent_prompt_runnable | agent_model
        
        # Create the graph
        workflow = StateGraph(AgentState, output=FinalResponse)
        
        # Add nodes
        workflow.add_node("generate_analysis_plan", self.generate_analysis_plan)
        workflow.add_node("react_agent", self.call_agent_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("generate_structured_response", self.generate_structured_response)
        
        # Set entry point
        workflow.set_entry_point("generate_analysis_plan")
        
        workflow.add_edge("generate_analysis_plan", "react_agent")
        # Add conditional edges
        workflow.add_conditional_edges(
            "react_agent",
            self.should_continue,
            path_map=["tools", "generate_structured_response"]
        )
        workflow.add_edge("tools", "react_agent")
        workflow.add_edge("generate_structured_response", END)
        
        # Compile the graph
        self.agent_graph = workflow.compile(
            debug=debug,
            name=self.name
        )
        
        return self