import argparse
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from audio_gemini_model import AudioGeminiModel
from prompt_manager import LlangChainPromptManager

# Try importing token_tracker from project root
try:
    from token_tracker import TokenTracker
except ImportError:
    TokenTracker = None
    print("Warning: token_tracker module not found. Token tracking will be disabled.")

logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("getFilesLogger")


def main():

    prompt_version_choices = [1, 2, 3, 4]
    most_recent_prompt = prompt_version_choices[-1]

    model_version_choices = [1, 2, 3]
    most_advanced_model = model_version_choices[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="screen",
        choices=["camera", "screen", "none"],
        help="Pixels to use",
    )
    parser.add_argument(
        "--model_type",
        type=int,
        default=most_advanced_model,
        choices=model_version_choices,
        help="Type of model to use (1 for thinking 2 for non-thinking)",
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=most_recent_prompt,
        choices=prompt_version_choices,
        help="The version of the prompt that is loaded on llangchain",
    )
    args = parser.parse_args()

    prompt_manager = LlangChainPromptManager(version=args.prompt_version)
    prompt_manager.load_prompt_name()
    prompt_manager.get_llang_chain_access()
    # Get the prompt template as text/str
    prompt_text = prompt_manager.prompt_template

    gem_ai_pgm_run = AudioGeminiModel(video_mode=args.mode)

    gem_ai_pgm_run.model_type = args.model_type
    print("-" * 50)

    print("...Current arguments used...")
    for k, v in args.__dict__.items():
        print(f"{k}: '{v}' ")
    print("-" * 50)
    gem_ai_pgm_run.prompt_version = args.prompt_version
    gem_ai_pgm_run.prompt = prompt_text

    asyncio.run(gem_ai_pgm_run.run())


if __name__ == "__main__":
    main()
