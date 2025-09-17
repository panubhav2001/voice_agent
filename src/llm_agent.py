import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=api_key)

def extract_identity(text: str) -> str:
    """
    Extracts identity information from text, always returning a JSON string.
    Uses null for fields that cannot be found.
    """
    # 1. The prompt is updated to always require a JSON output with null for missing values.
    system_prompt = """You are a precise information extraction tool. Your job is to extract the first name, last name, and year of birth from the user's text.

Your output MUST be a JSON object in the exact format: {"first_name": "...", "last_name": "...", "year_of_birth": ...}

❗IMPORTANT:
- If any piece of information is not found, you MUST use `null` as its value.
- Do NOT return any explanation or other text. Only the JSON object.

Here are some examples:
- Input: "My name is Alice Johnson and I was born in 1995." -> Output: {"first_name": "Alice", "last_name": "Johnson", "year_of_birth": 1995}
- Input: "Hello, I'm Maria. I was born in 1992." -> Output: {"first_name": "Maria", "last_name": null, "year_of_birth": 1992}
- Input: "John Doe 2004" -> Output: {"first_name": "John", "last_name": "Doe", "year_of_birth": 2004}
- Input: "I want to schedule a cleaning." -> Output: {"first_name": null, "last_name": null, "year_of_birth": null}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        # 2. This forces the model to output a syntactically correct JSON object.
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()

def extract_booking_details(text: str) -> str:
    """
    Extracts the desired day and time from user text using an LLM.
    """
    # Using the same Groq client as before
    system_prompt = """You are a precise data extraction tool. Your job is to extract the desired day of the week and time for an appointment from the user's text.

Your output MUST be a JSON object in the exact format: {"day": "...", "time": "..."}.

- The 'day' can be a specific day like 'Monday', 'Tuesday', or a relative day like 'today', 'tomorrow'.
- The 'time' should be in a clear format like '4:00 PM' or '11:00 AM'.
- If you cannot find a day or a time, use `null` for the respective value.

Examples:
- Input: "How about tomorrow at 4 in the afternoon" -> Output: {"day": "tomorrow", "time": "4:00 PM"}
- Input: "I'm free on Friday around 11am" -> Output: {"day": "Friday", "time": "11:00 AM"}
- Input: "tuesday morning" -> Output: {"day": "Tuesday", "time": "10:00 AM"}
- Input: "I don't know yet" -> Output: {"day": null, "time": null}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()