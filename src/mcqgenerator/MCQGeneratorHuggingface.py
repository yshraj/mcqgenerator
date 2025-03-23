import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import SequentialChain

# Load environment variables from the .env file
load_dotenv()

# Access the Hugging Face API token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load the Mistral-7B-Instruct-v0.2 model from Hugging Face API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    temperature=0.7,
    max_new_tokens=1024  # Increased to allow full JSON output
)

# Define the MCQ generation prompt
template = """
Text: {text}
You are an expert MCQ generator. Based on the text above, create {number} multiple-choice questions 
for {subject} students at a difficulty level of {tone}. 

Your response **MUST be in valid JSON format**. Follow these strict rules:
1. **DO NOT** include any explanations, markdown, or extra text—**return ONLY valid JSON**.
2. **Ensure JSON syntax is correct**, with proper quotes and formatting.
3. **Use unique, non-repetitive questions** strictly based on the text.
4. **Options should be diverse and not misleading**.
5. **Each question should have exactly four options (A, B, C, D)** and one correct answer.

### **JSON RESPONSE FORMAT**
```json
{response_json}"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template
)

# Define the MCQ generation chain
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

# Define the quiz evaluation prompt
template2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the question and give a complete analysis of the quiz. Use at most 50 words for complexity analysis. 
If the quiz is not appropriate for the cognitive and analytical abilities of the students,\
update the quiz questions and adjust the tone so that it perfectly fits the student's abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

# Define the quiz evaluation chain
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# Combine both chains into a SequentialChain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=False
)