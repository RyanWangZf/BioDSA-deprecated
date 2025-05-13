from langchain_core.prompts import PromptTemplate

FINAL_ANSWER_PROMPT_TEMPLATE = """
# TASK
Evaluate the user's scientific hypothesis taking into account the evidence provided.

# IMPORTANT
You should make your final answer completely based on the observations provided.
Do not make any assumptions or include any other information which is not included in the observations.

# OBSERVATIONS
{observations}

# FINAL ANSWER
The final answer is one of the following values:

True - the hypothesis is supported by the data
False - the hypothesis is not supported by the data
Not Verifiable - The hypothesis is not verifiable with the provided datasets

As a part of the final answer, you must output
- hypothesis: the hypothesis that you evaluated
- evidence: a list of evidence that justifies your evaluation of the hypothesis
- final_answer: one of the following values: True, False, Not Verifiable
"""

FINAL_ANSWER_PROMPT_TEMPLATE = PromptTemplate(
    template=FINAL_ANSWER_PROMPT_TEMPLATE,
    input_variables=["observations"]
)