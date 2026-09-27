import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def generate_response(prompt: str) -> str:
    """
    Send a prompt to Groq and return the generated response.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    return response.choices[0].message.content or ""