from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import openai
import os

app = FastAPI()

# Load OpenAI API Key from environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def fetch_wikipedia(topic):
    """Fetch a summary from Wikipedia."""
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join([p.text for p in paragraphs[:3]])

def summarize_text(text):
    """Summarize text using OpenAI's GPT."""
    if not OPENAI_API_KEY:
        return "OpenAI API Key is missing!"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Summarize the following text:"},
                  {"role": "user", "content": text}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"]

@app.get("/research/")
def research(topic: str):
    """API Endpoint for research"""
    raw_text = fetch_wikipedia(topic)
    summary = summarize_text(raw_text)
    return {"topic": topic, "summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
