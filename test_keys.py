import os 
from dotenv import load_dotenv 
load_dotenv() 
print("OpenAI Key:", "YES" if os.getenv("OPENAI_API_KEY") else "NO") 
print("Tavily Key:", "YES" if os.getenv("TAVILY_API_KEY") else "NO") 
