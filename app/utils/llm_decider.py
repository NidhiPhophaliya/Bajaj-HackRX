import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

# Load environment variables
load_dotenv(dotenv_path="D:/Bajaj-HackRX/.env")
api_key = os.getenv("GEMINI_API_KEY")

print("🧪 Loaded Gemini Key:", "Yes" if api_key else "❌ Not Found")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found. Gemini client will not be initialized.")

client = genai.GenerativeModel("gemini-2.5-flash") if api_key else None

def generate_decision(query, context_chunks):
    if client is None:
        raise ValueError("❌ Gemini client is not initialized.")

    prompt = f'''
Given the user query and policy clauses below, decide if the claim should be approved, estimate amount, and explain using clause references.

Query:
"{query}"

Relevant Clauses:
{chr(10).join(context_chunks)}

Respond in JSON like:
{{
  "decision": "approved | rejected",
  "amount": "<amount or N/A>",
  "justification": [
    {{ "clause": "<clause>", "reason": "<why>" }}
  ]
}}
'''

    print("🧠 Gemini Prompt Preview:\n", prompt[:500], "...\n")

    response = client.generate_content(prompt)
    raw_text = response.text

    # Clean output (in case it's wrapped in ```json ``` blocks)
    cleaned_text = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        if not cleaned_text:
            raise ValueError("❌ Gemini model returned an empty response.")
        parsed_json = json.loads(cleaned_text)
        print("✅ Gemini response parsed successfully")
        return parsed_json  # Return actual Python dict
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"🧾 Raw Gemini response: {raw_text}")
        raise
    except Exception as e:
        print(f"🔥 Unexpected error: {e}")
        print(f"🧾 Raw Gemini response: {raw_text}")
        raise
