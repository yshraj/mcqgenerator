# MCQ Generator using LangChain & Mistral-7B

## Overview
This project is an automated **Multiple-Choice Question (MCQ) Generator** that leverages **LangChain** and **Mistral-7B-Instruct-v0.2** (via Hugging Face) to create and evaluate quizzes based on provided text.

## Features
- Generates MCQs from input text using **Mistral-7B-Instruct**.
- Ensures that questions are **unique** and **contextually relevant**.
- Evaluates the complexity of generated MCQs.
- Adjusts the difficulty and tone for a target student audience.

## Tech Stack
- **Python** (Core language)
- **LangChain** (for LLM handling)
- **Mistral-7B-Instruct-v0.2** (via Hugging Face)
- **Hugging Face Inference API** (for model execution)
- **dotenv** (for environment variable management)

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yshraj/mcqgenerator
   cd mcq-generator
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your Hugging Face API key:
   ```sh
   HUGGINGFACE_TOKEN=your_huggingface_api_key
   ```

## Usage
streamlit run .\StreamlitAPP-huggingface.py

