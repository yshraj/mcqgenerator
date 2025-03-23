import os
import json
import traceback
import pandas as pd
import re
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging
from src.mcqgenerator.MCQGeneratorHuggingface import generate_evaluate_chain

#imporing necessary packages packages from langchain
from langchain.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI

import streamlit as st # type: ignore

with open("D:\\My-Projects\\mcqgenerator\\response.json") as file:
    RESPONSE_JSON = json.load(file)

# Streamlit UI
st.title("MCQ Generator from Text File 📄")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# User Inputs
num_mcqs = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5, step=1)
subject = st.text_input("Enter the Subject (e.g., Physics, Chemistry)")
complexity = st.selectbox("Select Complexity Level", ["Easy", "Medium", "Hard"])
button = st.button("Generate MCQs")

# Processing the file
if button and uploaded_file is not None and num_mcqs and subject and complexity:
    with st.spinner("loading..."):
        try:
            text = read_file(uploaded_file)
            
            with get_openai_callback() as cb:
                response=generate_evaluate_chain(
                    {
                        "text": text,
                        "number": num_mcqs,
                        "subject":subject,
                        "tone": complexity,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                    )
            
            st.subheader("Generated MCQs")
        except Exception as ex:
            print(ex)
            st.error('Error')

        else:
            if isinstance(response, dict):
                quizz = response.get("quiz",None)
                # Original text containing the JSON
                text = quizz

                # Regular expression to extract JSON content
                json_string = re.search(r'```json(.*?)```', text, re.DOTALL).group(1).strip()

                # Convert the string to a JSON object
                quiz = json.loads(json_string)
                quiz = json.dumps(quiz, indent=4)
                print(quiz)
                if(quiz is not None):
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index+1
                        st.table(df)
                        st.text_area(label="Review", value=response["review"])
                    else:
                        st.error("Error in table data")
            else:
                st.write(response)    

