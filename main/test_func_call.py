from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from get_files import getFiles
from audio_gemini_model import AudioLoop
from prompt_manager import LLangC_Prompt_Manager

# Define function declaration for `getFiles`
load_dotenv()
get_files_function = types.FunctionDeclaration(
    name="getFiles",
    description="Searches Google Drive for files by name or type, public or private based on OAuth.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "search_term": types.Schema(
                type=types.Type.STRING,
                description="Filename or file type to search (e.g., csv, pdf, report)",
            ),
            "use_oauth": types.Schema(
                type=types.Type.BOOLEAN,
                description="True for private files (OAuth), False for public files only",
            ),
        },
        required=["search_term", "use_oauth"],
    ),
)

prompt_manager = LLangC_Prompt_Manager()
prompt_manager.load_prompt_name()
prompt_manager.get_llang_chain_access()
# Get the prompt template as text/str
prompt_text = prompt_manager.prompt_template

gem_ai_pgm_run = AudioLoop()
tools = [
    types.Tool(code_execution=types.ToolCodeExecution),
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(function_declarations=[get_files_function]),
]

config = types.GenerateContentConfig(tools=tools)
gem_ai_pgm_run.create_client()

import pdb

pdb.set_trace()
# Send request with function declarations
response = gem_ai_pgm_run.client.models.generate_content(
    model="models/gemini-live-2.5-flash-preview",
    contents="Find me all csv files on my Google Drive.",
    config=config,
)

# Check for a function call
function_call_part = response.candidates[0].content.parts[0]
if function_call_part.function_call:
    function_call = function_call_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")

    # Call the actual function here:
    result = getFiles(**function_call.args)
    print("RESULT", result)
else:
    print("No function call found in the response.")
    print(response.text)
