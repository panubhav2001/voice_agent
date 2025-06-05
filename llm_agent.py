import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=api_key)

def extract_identity(text):
    prompt = f"""
You are an information extractor. Your job is to extract the first name, last name, and year of birth from a sentence or phrase.

Your output MUST be either:
- A **JSON object** in the exact format: {{"first_name": "...", "last_name": "...", "year_of_birth": 1234}}
- OR the string: "Invalid"

❗IMPORTANT: Do NOT return any explanation, do NOT include Python code or any other text. Only return the JSON or "Invalid".

Here are some examples:

Input: "My name is Alice Johnson and I was born in 1995."
Output: {{"first_name": "Alice", "last_name": "Johnson", "year_of_birth": 1995}}

Input: "This is John Doe, born in 1987."
Output: {{"first_name": "John", "last_name": "Doe", "year_of_birth": 1987}}

Input: "Hello, I'm Maria. I was born in 1992."
Output: "Invalid"

Input: "John Doe 2004"
Output: {{"first_name": "John", "last_name": "Doe", "year_of_birth": 2004}}

Input: "Michael 1998"
Output: "Invalid"

Now process the following input:

Input: "{text}"
Output:
"""


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
