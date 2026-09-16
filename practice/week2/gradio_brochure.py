import os
from dotenv import load_dotenv
from typing import Literal
from gradio import Markdown
from networkx import display
from openai import OpenAI
from typing import Literal
import json
from practice.reusable.webscrapper import scrape_any_site_selenium
from IPython.display import Markdown, display, update_display

load_dotenv()


AI_BASE_URL = os.getenv('AI_BASE_URL', "https://generativelanguage.googleapis.com/v1beta/openai/")
# change the url to http://localhost:11434/v1 for use local ollama i have gemma 2b but i dont gonna use local as i dont need as of now

API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('OLLAMA_API_KEY')
# MODEL_NAME = "gemini-3.5-flash-lite"
MODEL_NAME = "gemma4:31b"

URL_EXTRACTION_SYSTEM_PROMPT = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def found_relevant_links(url):
    print("founding relevant links for the url: ", url)
    ai_ready_text, scraped_links = scrape_any_site_selenium(url)
    user_prompt = f"""
    Here is the list of links on the website {url} -
    Please decide which of these are relevant web links for a brochure about the company, 
    respond with the full https URL in JSON format.
    Do not include Terms of Service, Privacy, email links.
    
    Links (some might be relative links):
    
    """
    user_prompt += "\n".join(scraped_links)
    ollama = OpenAI(base_url=AI_BASE_URL, api_key=API_KEY)
    response = ollama.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": URL_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    result = result.strip()

    if result.startswith("```"):
        result = result.removeprefix("```json").removeprefix("```")
        result = result.removesuffix("```").strip()
    links = json.loads(result)
    print(f"Found {len(links['links'])} relevant links")
    return links
    

def fetch_page_and_all_relevant_links(url):
    contents = scrape_any_site_selenium(url)[0]
    relevant_links = found_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += scrape_any_site_selenium(link["url"])[0]
    return result

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
        You are looking at a company called: {company_name}
        Here are the contents of its landing page and other relevant pages;
        use this information to build a short brochure of the company in markdown without code blocks.\n\n
    """
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:10_000] # Truncate if more than 10,000 characters
    return user_prompt

BROCHURE_SYSTEM_PROMPT = """
    You are an assistant that analyzes the contents of several relevant pages from a company website
    and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
    Respond in markdown without code blocks.
    Include details of company culture, customers and careers/jobs if you have the information.
"""

def create_brochure(company_name, url, model_name):
    
    ollama = OpenAI(base_url=AI_BASE_URL, api_key=API_KEY)

    response = ollama.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    # print(result)
    return result


def stream_brochure(company_name, url, model_name):
    
    ollama = OpenAI(base_url=AI_BASE_URL, api_key=API_KEY)

    response = ollama.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        response_text += content
        # print(content, end="", flush=True)
        yield response_text


import gradio as gr
def generate(company_name, url, action, model_name):
    if action == "Create Brochure":
        yield create_brochure(company_name, url, model_name)
    else:
        yield from stream_brochure(company_name, url, model_name)

model_list = ["gpt-oss:120b", "gemma4:31b", "minimax-m3"]

with gr.Blocks() as demo:
# gr.Markdown("# My Custom Chatbot")

    with gr.Row():
        with gr.Column():
              
            # gr.Label("Enter your prompt below and click 'Generate magic' to get a response from the chatbot.", size="sm")
            out = gr.Markdown(label="Output Response")
            inp = gr.Textbox(placeholder="Enter company name...", label="Company Name")
            url = gr.Textbox(placeholder="Enter company URL...", label="Company URL")
            dd = gr.Dropdown(choices=["Create Brochure", "Stream Brochure"], label="Select Action", value="Create Brochure")
            btn = gr.Button("Generate magic", variant="primary",size="sm", elem_id="generate-btn")
            model_dd = gr.Dropdown(choices=model_list, label="Select Model", value="gpt-oss:120b") 
    btn.click(fn=generate,
              inputs=[inp, url, dd, model_dd], outputs=out)


demo.launch()
