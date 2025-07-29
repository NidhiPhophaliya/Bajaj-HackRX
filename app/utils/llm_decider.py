import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

# Load environment variables from the specified .env file
load_dotenv(dotenv_path="D:/Bajaj-HackRX/.env")

# Retrieve the API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

# Print a status message indicating if the API key was loaded successfully
print("🧪 Loaded Gemini Key:", "Yes" if api_key else "❌ Not Found")

# Configure the Gemini API with the retrieved API key.
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found. Gemini client will not be initialized.")

# Initialize the GenerativeModel client.
client = genai.GenerativeModel("gemini-2.5-flash") if api_key else None

def generate_decision(query, context_chunks):
    """
    Generates a decision (approved/rejected), estimated amount, and justification
    for an insurance claim based on a user query and relevant policy clauses.

    Args:
        query (str): The user's claim query.
        context_chunks (list): A list of relevant policy clauses (strings).

    Returns:
        str: A JSON string containing the decision, amount, and justification.

    Raises:
        ValueError: If the Gemini client is not initialized (e.g., API key is missing).
        json.JSONDecodeError: If the model's response is not valid JSON.
    """
    if client is None:
        raise ValueError("❌ Gemini client is not initialized. API key is missing.")

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

    response = client.generate_content(prompt)
    raw_text = response.text

    # Remove Markdown code block markers if present
    cleaned_text = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        if not cleaned_text:
            raise ValueError("❌ Gemini model returned an empty response.")
        parsed_json = json.loads(cleaned_text)
        return cleaned_text
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Raw response text: '{raw_text}'")
        raise json.JSONDecodeError(
            f"❌ Gemini model did not return valid JSON. Error: {e}. Raw response: '{raw_text}'",
            e.doc, e.pos
        )
    except Exception as e:
        print(f"An unexpected error occurred while processing Gemini response: {e}")
        print(f"Raw response text: '{raw_text}'")
        raise

# Example usage (uncomment to test):
# if __name__ == "__main__":
#     if client:
#         sample_query = "My car was damaged in a collision with another vehicle."
#         sample_clauses = [
#             "Clause 101: Collision damage is covered up to $5000.",
#             "Clause 102: Damage due to natural disasters is not covered.",
#             "Clause 103: Deductible of $500 applies to all collision claims."
#         ]
#         try:
#             decision_json = generate_decision(sample_query, sample_clauses)
#             print("\nGenerated Decision:")
#             print(decision_json)
#         except ValueError as e:
#             print(f"Error: {e}")
#         except json.JSONDecodeError as e:
#             print(f"JSON Decode Error: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#     else:
#         print("Cannot run example: Gemini client not initialized.")