from pydantic import BaseModel, Field
from typing import List, Literal, Annotated, Sequence, TypeVar, TypedDict

from langgraph.graph.message import add_messages, BaseMessage
from langgraph.managed import IsLastStep, RemainingSteps


class UserRequest(TypedDict):
    input_query: str
    dataset_paths: str
    dataset_schema: str

class AnalysisPlan(BaseModel):
    input_query: str
    plan: str

"""
Used for returning back to the user. includes additional information that is not needed for the structured output.
"""
class FinalResponse(BaseModel):
    hypothesis: str;
    final_answer: str;
    code: str;
    language: Literal["python", "r", "sas"];
    analysis_plan: str;
    observations: str;
    evidence: List[str];
    def __str__(self):
        return f"Hypothesis: {self.hypothesis}\nFinal Answer: {self.final_answer}\nEvidence: {"\n".join(self.evidence)}"


class FinalResponseForStructuring(BaseModel):
    """
    Used for generating the final response from the structured output of the model.
    """
    final_answer: str;
    evidence: List[str];
    def __str__(self):
        return f"Final Answer: {self.final_answer}\n Observations: {self.observations}"


class StructuredAnswer(BaseModel):
    final_answer: Literal["True", "False", "Not Verifiable"];
    evidence: List[str];
    def __str__(self):
        return f"Final Answer: {self.final_answer}\nEvidence: {"\n".join(self.evidence)}"

class CodeResult(BaseModel):
    input_query: str;
    plan: str;
    code: str;
    execution_results: str;

    def __str__(self):
        return f"Input Query: {self.input_query}\nPlan: {self.plan}\nCode: {self.code}\nLanguage: {self.language}\nExecution Results: {self.execution_results}"

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    structured_response: FinalResponse = None


class ReActState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep
    input_tokens: int # the number of input tokens used
    output_tokens: int # the number of output tokens used


StateSchema = TypeVar("StateSchema", bound=AgentState)