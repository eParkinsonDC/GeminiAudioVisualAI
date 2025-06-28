import asyncio
import base64
import io
import os
import re
import shutil
import time
import traceback
import websockets

import cv2
import mss
import PIL.Image
from google import genai
from google.genai import types

# Group all main.* imports together
try:
    from get_files import getFiles
except ImportError:
    getFiles = None


try:
    from prompt_manager import LlangChainPromptManager
except ImportError:
    LlangChainPromptManager = None


try:
    from token_tracker import TokenTracker
except ImportError:
    TokenTracker = None

try:
    from config import (
        PYA,
        FORMAT,
        CHANNELS,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
        OUTPUT_DIR,
        OUTPUT_FILE,
        CHUNK_SIZE,
    )
except ImportError:
    PYA = None
    FORMAT = None
    CHANNELS = None
    SEND_SAMPLE_RATE = None
    RECEIVE_SAMPLE_RATE = None
    OUTPUT_DIR = None
    OUTPUT_FILE = None
    CHUNK_SIZE = None


class AudioGeminiModel:
    """
    AudioGeminiModel  manages real-time audio and video interaction with
    a Gemini AI endpoint.

    This class encapsulates the setup and management of audio input/output,
    video or screen capture, and asynchronous communication with a Gemini AI
    model for live conversational experiences.

    It supports sending and receiving audio, handling text prompts,
    managing session state, and
    optionally capturing video frames or screen images for multimodal input.

    Attributes:
        video_mode (str): Determines the video input mode
        ("camera" or "screen").

        audio_in_queue (asyncio.Queue): Queue for incoming
        audio data to be played.

        out_queue (asyncio.Queue): Queue for outgoing data (audio, video,
        or screen frames).

        session: The active Gemini live session object.

        client: The Gemini API client instance.

        config: The configuration object for the Gemini model session.

        model_type (int): Selects the Gemini model variant to use.

        model (str): The model identifier string.

        last_active (float): Timestamp of the last user or system activity.

        awaiting_response (bool): Indicates if the system is
        waiting for a response.

        unanswered_prompts (int): Counter for consecutive unanswered
        keep-alive prompts.

        max_unanswered (int): Maximum allowed unanswered
        prompts before pausing.

        prompt (str): The system instruction or prompt for the AI model.

        prompt_version: Version identifier for the prompt.

        prompt_manager: Optional prompt management utility.

        token_tracker: Optional token usage tracking utility.
    Methods:

        create_client(): Initializes the Gemini API client if not already set.

        create_config(): Builds the configuration for the Gemini model session.

        create_model(): Selects and sets the Gemini model based on model_type.

        strip_code_blocks(text): Removes code block formatting from text.
        save_code_to_file(code, filename): Saves code snippets to a file.


        send_text(): Asynchronously reads user text input and
        sends it to the session.

        _get_frame(cap): Captures and processes a video frame from a camera.

        keep_alive(interval, idle_threshold): Periodically sends keep-alive
        prompts if idle.

        get_frames(): Asynchronously captures and queues video
        frames from the camera.

        _get_screen(): Captures and processes a screenshot of the display.
        get_screen(): Asynchronously captures and queues screen images.

        send_realtime(): Sends queued data (audio/video/screen) to the
        session in real time.

        listen_audio(): Asynchronously streams microphone audio to
        the output queue.

        receive_audio(): Receives and processes audio and text responses
        from the session.

        play_audio(): Asynchronously plays audio data from the input queue.

        run(): Main entry point to start the audio loop and manage
        all asynchronous tasks.
    Usage:
        Instantiate AudioLoop, configure the prompt and model as needed,
        and call `await run()`
        within an asyncio event loop to start the interactive session.
    """

    def __init__(self, video_mode="screen"):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.client = None
        self.config = None
        self.output_file_path = OUTPUT_FILE
        self.model_type = 1
        self.model = None
        self.last_active = time.time()
        self.uploaded_files = None
        self.awaiting_response = False
        self.unanswered_prompts = 0
        self.max_unanswered = 3
        self.prompt = None
        self.prompt_version = None
        self.session_handle = None
        self.handle_path = None
        self.latest_session_handle = self.session_handle
        self.prompt_manager = (
            LlangChainPromptManager() if LlangChainPromptManager else None
        )
        self.token_tracker = TokenTracker() if TokenTracker else None

    def create_client(self):
        """
        Initializes the genai.Client instance if it does not already exist.

        This method checks if the `client` attribute is None, and if so,
        creates a new
        `genai.Client` object using the API key from the environment variable
        'GEMINI_API_KEY' and sets the HTTP API version to 'v1beta'.

        Raises:
            Any exceptions raised by the genai.Client constructor
            if initialization fails.
        """
        if self.client is None:
            self.client = genai.Client(
                http_options={"api_version": "v1beta"},
                api_key=os.environ.get("GEMINI_API_KEY"),
            )

        return True

    def sanitize_name(self, filename):
        # Remove extension, convert to lowercase, replace non-allowed
        # chars with dash, strip dashes
        name = os.path.splitext(filename)[0].lower()
        name = re.sub(
            r"[^a-z0-9-]", "-", name
        )  # only allow lowercase letters, numbers, dash
        name = name.strip("-")
        return name or "file"  # fallback

    async def upload_all_files(self, files_dir="../files"):
        os.makedirs(files_dir, exist_ok=True)

        # --- Delete all files from Gemini client (ASYNC) ---
        try:
            print("Listing files on client for deletion...")
            pager = await self.client.aio.files.list()

            async for f in pager:
                try:
                    await self.client.aio.files.delete(name=f.name)
                    print(f"Deleted file: {f.name}")
                except Exception as e:
                    print(f"Failed to delete file {f.name}: {e}")
        except Exception as e:
            print(f"Failed to list files on client: {e}")

        tasks = []
        for filename in os.listdir(files_dir):
            file_path = os.path.join(files_dir, filename)
            if os.path.isfile(file_path):
                # Schedule the async upload
                tasks.append(self.client.aio.files.upload(file=file_path))
        # Await all uploads to finish
        results = await asyncio.gather(*tasks)
        print("All uploads complete!")

        return results

    def create_config(self, files=None):
        """
        Creates and sets the configuration for live audio interaction,
        optionally using uploaded files for retrieval.
        """
        if not self.prompt:
            raise ValueError("Prompt must be set before calling create_config()")
        if files is None:
            files = getattr(self, "uploaded_files", [])

        # If you want to use the files for retrieval (if API supports)
        retrieval_sources = []
        for f in files:
            # Depending on API: may need f.name, f.uri, or f.resource_name
            retrieval_sources.append(types.File(name=f.name, mime_type="csv"))
            # or types.RetrievalSource(uri=f.uri) etc

        if not self.prompt:
            raise ValueError("Prompt must be set before calling create_config()")

        tools_simplified = [
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(code_execution=types.ToolCodeExecution()),
        ]

        # tools_ext = [
        #     types.Tool(google_search=types.GoogleSearch()),
        #     types.Tool(code_execution=types.ToolCodeExecution()),
        #     types.Tool(
        #         function_declarations=[
        #             types.FunctionDeclaration(
        #                 name="getFiles",
        #                 description="Searches Google Drive for files by search term name/type",
        #                 parameters=types.Schema(
        #                     type=types.Type.OBJECT,
        #                     properties={
        #                         "search_term": types.Schema(
        #                             type=types.Type.STRING,
        #                             description="Filename or file type (e.g., csv, pdf, docx, John, Report, etc.).",
        #                         ),
        #                     },
        #                     required=["search_term"],
        #                 ),
        #             ),
        #         ]
        #     ),
        # ]

        # TODO: Fix and ensure file retrieval works with extended tools

        if self.model_type > 1:
            # NOTE: tools_ext is problematic at the moment with current models

            tools = tools_simplified
        else:
            tools = tools_simplified

        # Read saved handle if exists
        handle_path = os.path.join(
            os.path.dirname(__file__), "..", OUTPUT_DIR, "session_handle.txt"
        )
        self.handle_path = handle_path

        if os.path.exists(self.handle_path):
            with open(self.handle_path, "r", encoding="utf-8") as f:
                session_handle = f.read().strip()

            self.session_handle = session_handle
            print(f"Loaded session handle: {self.session_handle}")
        else:
            print("No saved session handle found.")
        self.config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Sulafat"
                    )
                )
            ),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ALL_INPUT"
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            tools=tools,  # Correctly formatted list of tools
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.prompt)], role="user"
            ),
            session_resumption=types.SessionResumptionConfig(handle=None),
        )

        return True

    def create_model(self):
        """
        Selects and sets the appropriate model based on the instance's model_type attribute.

        The method uses a predefined mapping of model types to model paths. If the model_type
        is recognized, it assigns the corresponding model path to self.model. If the model_type
        is not recognized, it raises a ValueError.

        Returns:
            bool: True if the model was successfully set.

        Raises:
            ValueError: If the model_type is not found in the model_map.
        """
        model_map = {
            # Version 1 - Thinking model
            1: "models/gemini-2.5-flash-exp-native-audio-thinking-dialog",
            # Version 2 - Non-thinking model/ Has access to function calling
            2: "models/gemini-2.5-flash-preview-native-audio-dialog",
            # Version 3 - Has access to all but suffers in performance
            3: "models/gemini-live-2.5-flash-preview",
        }
        self.model = model_map.get(self.model_type)
        if not self.model:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        print(f'Using model: {self.model.split("/")[-1]}')
        return True

    def strip_code_blocks(self, text):
        if not text:
            return ""
        # Remove opening triple backticks but keep optional language name
        text = re.sub(r"```([a-zA-Z]*)\n", r"\1\n", text)
        # Remove any other triple backticks
        text = text.replace("```", "")
        return text

    async def send_text(self):
        """
        Asynchronously prompts the user for text input in a loop and sends the input to the session.

        The loop continues until the user enters a quit command ("q", "quit", or "exit").
        Each input is sent to the session using `self.session.send()`, with an empty input replaced by a period (".").
        Updates `self.last_active` with the current time after each input.

        Returns:
            None
        """
        quit_commands = {"q", "quit", "exit"}
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.strip().lower() in quit_commands:
                print("\n   Quit command received. Ending session...")
                break
            self.last_active = time.time()
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        """
        Captures a single frame from the given video capture object, processes it, and returns a dictionary containing the JPEG-encoded image data in base64 format along with its MIME type.

        Args:
            cap (cv2.VideoCapture): OpenCV video capture object from which to read the frame.

        Returns:
            dict or None: A dictionary with keys 'mime_type' (str) and 'data' (str, base64-encoded JPEG image) if a frame is successfully captured; otherwise, None.
        """
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def keep_alive(self, interval=60, idle_threshold=90):
        """
        Periodically checks for user inactivity and sends keep-alive prompts
        to maintain engagement.

        Args:
            interval (int, optional): Time in seconds between each
            inactivity check. Defaults to 60.
            idle_threshold (int, optional): Time in seconds of inactivity
            before sending a prompt. Defaults to 90.

        Behavior:
            - If the user has been inactive for longer than `idle_threshold`
            and a session exists,
            sends a keep-alive prompt ("Are you still there?").
            - If multiple prompts go unanswered (as determined by
            `self.max_unanswered`), sends a follow-up message
            ("I'll pause until you're back. Just say anything to continue.")
            and resets the unanswered prompt counter.
            - Handles exceptions during message sending
            and prints warning messages.
            - Updates activity and prompt tracking attributes accordingly.
        """
        while True:
            await asyncio.sleep(interval)
            time_since_active = time.time() - self.last_active
            if time_since_active > idle_threshold and self.session:
                if not self.awaiting_response:
                    try:
                        prompt = "Are you still there?"
                        print(f"\n Asking: {prompt}\n")
                        await self.session.send(input=prompt, end_of_turn=True)
                        self.awaiting_response = True
                        self.unanswered_prompts += 1
                        self.last_active = time.time()
                    except Exception as e:
                        print(f"Keep-alive failed: {e}")
                elif self.unanswered_prompts >= self.max_unanswered:
                    try:
                        follow_up = "I'll pause until you're back. Just say anything to continue."
                        print(f"\n Follow-up: {follow_up}\n")
                        await self.session.send(input=follow_up, end_of_turn=True)
                        self.awaiting_response = True
                        self.unanswered_prompts = 0  # Reset so it doesn't loop forever
                        self.last_active = time.time()
                    except Exception as e:
                        print(f" Follow-up failed: {e}")

    async def get_frames(self):
        """
        Asynchronously captures video frames from the default camera device and puts them into an output queue.

        This coroutine initializes a video capture object in a separate thread, then continuously retrieves frames
        using a helper method (`_get_frame`) also executed in a thread. Each valid frame is put into `self.out_queue`
        after a 1-second delay. The loop terminates when no more frames are available, and the video capture resource
        is released.

        Yields:
            None. Frames are put into `self.out_queue` for consumption elsewhere.

        Raises:
            Any exceptions raised by `cv2.VideoCapture` or frame retrieval will propagate unless handled elsewhere.
        """
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        """
        Captures a screenshot of the primary monitor, converts it to JPEG format, and returns the image data encoded in base64.

        Returns:
            dict: A dictionary containing:
                - "mime_type" (str): The MIME type of the image ("image/jpeg").
                - "data" (str): The base64-encoded JPEG image data.
        """
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def get_screen(self):
        """
        Asynchronously captures screen frames in a loop and puts them into an output queue.

        Continuously retrieves screen frames by running the synchronous `_get_screen` method in a separate thread.
        If a frame is `None`, the loop breaks. Otherwise, the frame is put into `self.out_queue` after a 1-second delay.

        Returns:
            None
        """
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        try:
            while True:
                msg = await self.out_queue.get()
                await self.session.send(input=msg)
        except websockets.exceptions.ConnectionClosedError as e:
            print("\n[Zelda] 🔌 Lost connection to Gemini server! Reason:", str(e))
            print("[Zelda] Please check your network and restart the app.")
        except Exception as e:
            print(f"[Zelda] Unexpected error in send_realtime: {e}")

    async def listen_audio(self):
        """
        -Asynchronously listens to audio input from the default microphone
            and streams audio data chunks to an output queue.
        - This method opens an audio stream using the default input device
            and continuously reads audio data in chunks.
        - Each chunk is placed into an asynchronous output queue along with
            its MIME type.
        -The method also updates the timestamp of
            the last activity after each read.
        -Designed to run in an asynchronous event loop.

        Raises:
            Any exceptions raised by the underlying
            audio library or asyncio operations.
        """
        mic_info = PYA.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            PYA.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            self.last_active = time.time()
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    def clear_output_file(self, filename=None):
        """
        Clears the output file specified by filename or by self.output_file_path if not provided.
        """
        if filename is None:
            filename = self.output_file_path
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR)
        )
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                pass  # This just truncates the file
            print(f"Cleared output file at {filepath}")
        except Exception as e:
            print(f"Failed to clear file: {e}")

    def save_code_to_file(self, code, filename=None, mode="a"):
        if filename is None:
            filename = self.output_file_path
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR)
        )
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Read the last character in the file to decide on spacing (only if file exists and is not empty)
        last_char = ""
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, "rb") as f:
                f.seek(-1, os.SEEK_END)
                last_char = f.read(1).decode("utf-8", errors="ignore")

        # If last character is not a space or newline, insert a space
        needs_space = last_char not in {"", " ", "\n"}

        code = code.strip()
        if not code:
            return

        code_to_write = ""
        if needs_space:
            code_to_write += " "

        code_to_write += code

        # Only append a newline if the string ends with . ? or !
        if re.search(r"[.?!]$", code):
            code_to_write += "\n"

        try:
            with open(filepath, mode, encoding="utf-8") as f:
                f.write(code_to_write)
        except Exception as e:
            print(f"Failed to save code: {e}")

    def get_all_lines_from_output(self, filename=None):
        # Default to self.output_file_path if not provided
        if filename is None:
            filename = self.output_file_path

        # If filename is not absolute, use outputs directory
        if not os.path.isabs(filename):
            output_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR)
            )
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename

        if not os.path.exists(filepath):
            print(f"No output file found at {filepath}")
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]
            term_width = shutil.get_terminal_size(
                (80, 20)
            ).columns  # Default to 80 if unknown

            # Color codes
            color = "\033[36m"  # Cyan
            reset = "\033[0m"

            print(
                f"\n{color}{'-------------------------------------- Start of Response --------------------------------------'.center(term_width)}{reset}\n"
            )
            for n, line in enumerate(lines):
                print(f"{color}{str(n+1).rjust(3)}. {line.center(term_width-7)}{reset}")
            print(
                f"\n{color}{'-------------------------------------- End of Response --------------------------------------'.center(term_width)}{reset}\n"
            )
            return lines
        except Exception as e:
            print(f"Failed to read file: {e}")
            return []

    async def receive_audio(self):
        try:
            while True:
                turn = self.session.receive()
                last_text = None  # For final, if needed
                async for response in turn:
                    # === 1. Function/Tool Call Handling ===
                    tool_calls = getattr(response, "tool_calls", None)
                    if tool_calls:
                        for tool_call in tool_calls:
                            fn_name = tool_call["name"]
                            arguments = tool_call.get("arguments", {})
                            print(f"\n[Gemini Tool Call] {fn_name}({arguments})")
                            if fn_name == "getFiles":
                                result = getFiles(**arguments)
                            else:
                                result = {
                                    "success": False,
                                    "error": f"Unknown tool: {fn_name}",
                                }
                            # --- Return result to Gemini ---
                            await self.session.send(
                                tool_response={
                                    "name": fn_name,
                                    "result": result,
                                },
                                end_of_turn=True,
                            )
                            continue  # Don't process rest of this response, move to next
                        continue  # Move to next response after handling all tool calls

                    # === 2. Look for Session Handler (Allowing resuming session) ===
                    if getattr(response, "session_resumption_update", None):
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            print(f"[Session updated] new handle: {update.new_handle}")
                            self.latest_session_handle = update.new_handle

                            with open(self.handle_path, "w", encoding="utf-8") as f:
                                f.write(self.latest_session_handle)

                    # === 3. Usage tracking ===
                    if (
                        self.token_tracker
                        and hasattr(response, "usage_metadata")
                        and response.usage_metadata
                    ):
                        self.token_tracker.add_usage(response.usage_metadata)
                        print(self.token_tracker.summary())

                    # === 4. Audio chunk for playback ===
                    if hasattr(response, "data") and response.data:
                        self.audio_in_queue.put_nowait(response.data)
                        continue

                    # === 5. Gemini output_transcription as text ===
                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        output_trans = getattr(
                            server_content, "output_transcription", None
                        )
                        if output_trans:
                            text = getattr(output_trans, "text", None)
                            if text:
                                last_text = text
                                self.save_code_to_file(
                                    text, filename=self.output_file_path, mode="a"
                                )
                            continue

                    # === 6. Fallback: plain text (for non-audio responses) ===
                    if hasattr(response, "text") and response.text:
                        print(response.text, end="")
                        self.save_code_to_file(
                            response.text, filename=self.output_file_path, mode="a"
                        )

                # === 7. End-of-turn display ===
                if last_text or getattr(response, "turn_complete", None):
                    self.get_all_lines_from_output()

                # === 8. Clear input audio queue after turn ===
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

        except asyncio.CancelledError:
            print("receive_audio cancelled cleanly.")
        except Exception as e:
            print(f"Error in receive_audio: {e}")

    async def play_audio(self):
        """
        Asynchronously plays audio data from the audio input queue.

        This coroutine initializes an audio output stream in a separate thread using the specified
        audio format, channel count, and sample rate. It then enters an infinite loop, where it
        continuously retrieves audio byte streams from `self.audio_in_queue` and writes them to
        the output stream, also in a separate thread, to avoid blocking the event loop.

        Raises:
            Any exceptions raised by the audio stream initialization or writing process.
        """
        stream = await asyncio.to_thread(
            PYA.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """
        Asynchronously runs the main audio loop, managing the lifecycle of the audio client, model, and configuration.
        This method sets up the client, model, and configuration if they are not already initialized. It then establishes a live session
        and creates multiple asynchronous tasks for handling text sending, real-time audio processing, audio listening, video frame or screen capture,
        audio receiving, audio playback, and keep-alive signaling. The method waits for the text sending task to complete before raising a
        cancellation error to terminate the loop. Handles task group exceptions and ensures proper cleanup of audio streams.
        Raises:
            asyncio.CancelledError: When the user requests exit or the main task is cancelled.
            ExceptionGroup: If any exceptions occur within the task group, they are handled and printed.
        """
        try:
            self.clear_output_file(self.output_file_path)
            self.create_client()
            self.uploaded_files = await self.upload_all_files()
            if self.model is None:
                if self.create_model():
                    print("Model is created.")
            if self.config is None:
                if self.create_config():
                    print("Model config is setup.")

            async with (
                self.client.aio.live.connect(
                    model=self.model, config=self.config
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                initial_handle = getattr(session, "session_resumption_handle", None)
                if initial_handle:
                    print(f"[Session] initial handle: {initial_handle}")
                    self.session_handle = initial_handle
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                if self.session_handle:
                    print(f"Saving session handle: {self.session_handle}")

                    with open(self.handle_path, "w", encoding="utf-8") as f:
                        f.write(self.session_handle)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.keep_alive(interval=60, idle_threshold=90))

                await send_text_task
                raise asyncio.CancelledError("User requested exit")
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            self.audio_stream.close()
            for e in eg.exceptions:
                print(f"TaskGroup error: {e}")
            traceback.print_exception(eg)
