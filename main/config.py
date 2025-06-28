import os
from dotenv import load_dotenv
import pyaudio

load_dotenv()  # Load variables from .env into environment

# Audio config
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Secret/API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Any other config values you want!
OUTPUT_DIR = "outputs"
OUTPUT_FILE = "output_from_ai.txt"

PYA = pyaudio.PyAudio()
SERVICE_ACCOUNT_FILE = "service_account.json"  # Path to your service account key file
