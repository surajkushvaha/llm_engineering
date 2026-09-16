from transformers import pipeline
# imports
from dotenv import load_dotenv

import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch

audio_filename = "D:\\1playground\\llm_engineering\\practice\\denver_extract.mp3"
audio_file = open(audio_filename, "rb")
load_dotenv(override=True)

hf_token = os.environ.get('HF_TOKEN')
print("Hugging Face token retrieved successfully.", hf_token)
login(hf_token, add_to_git_credential=True)
print("Logged in to Hugging Face Hub successfully.")
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    # device='cuda',
    return_timestamps=True
)

result = pipe(audio_filename)
transcription = result["text"]
print(transcription)