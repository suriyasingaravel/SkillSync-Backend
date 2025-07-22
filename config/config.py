# config.py
import os
from dotenv import load_dotenv

# 1. immediately load the .env file
load_dotenv()

# 2. now read the vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 3. error out only if *either* is missing
if not OPENAI_API_KEY or not GROQ_API_KEY:
    raise EnvironmentError("Missing required environment variables")
