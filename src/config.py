import os
import streamlit as st

# Load API key and passcode from secrets
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]
PASSCODE = st.secrets["general"]["PASSCODE"]

ENABLE_AUTH = False  # Set to False to disable authentication

# Hugging Face model repository
MODEL_NAME = "ksj3768/literary-roberta"

# Ensure API key is available
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OpenAI API Key is missing! Set it in the .env file.")

# Model configurations
MODEL_PATH = os.path.join("models", "Trained_Roberta.pth")
AUTHOR_MAP_PATH = os.path.join("assets", "author_map.json")

# Text processing
MAX_INPUT_TOKENS = 500  # Adjust based on OpenAI token limit
MAX_GENERATED_TOKENS = 500