# GeminiAudioAI

Audio/Visual AI using Gemini Model (Gemini-2.5) to use as a coding buddy for any Python, tech and/or Data Science tasks.
Supports multiple languages, screen sharing/capture, voice interaction, and prompt versioning.

---

## 🚀 Installation

### Poetry (Recommended)

1. **Install Poetry:**
    ```bash
    pip install poetry
    ```

2. **Install project dependencies:**
    ```bash
    poetry install
    ```

3. **Activate the virtual environment:**
    ```bash
    poetry shell
    # or manually (Windows): .venv\Scripts\activate
    # or manually (Linux/Mac): source .venv/bin/activate
    ```

4. **[Optional] Export requirements.txt for pip:**
    ```bash
    poetry self add poetry-plugin-export    # one-time
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

---

### Pip (Alternative)

1. **Install dependencies:**
    ```bash
    python -m venv .venv
    # Activate your environment:
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration

1. **Copy `.env.example` to `.env` and set your required keys:**
    - `GEMINI_API_KEY`
    - `LANGSMITH_API_KEY`
    - (Any others needed for your workflow)

2. **Check and edit `main/config.py` if you want to override default constants (e.g., audio device, chunk size, output paths, etc).**

---

## 🏃‍♂️ Running the App

```bash
# From project root
python -m main.main --mode screen --model_type 1 --prompt_version 2
```

---

## Screen Mode, Model, and Prompt Version

```bash
python main.py --mode screen --model_type 1 --prompt_version 2
```

---

## ✅ Enhancements Integrated

| Feature                                         | ✅ Implemented |
|-------------------------------------------------|---------------|
| **Monitor 1 selection** (avoids black screen)   | ✅ `sct.monitors[1]` fallback to `[0]` |
| **PNG format for clarity**                      | ✅ `cv2.imencode(".png", ...)` |
| **Skip PIL, use OpenCV directly**               | ✅ No PIL in screen capture path |
| **Sharpening filter via OpenCV**                | ✅ `cv2.filter2D` kernel |
| **RMS-based voice detection**                   | ✅ `_rms()` + threshold (300) |
| **"Are you still there?" prompt**               | ✅ `keep_alive()` logic |
| **Voice response detection (STT)**              | ✅ Detected via transcription in `receive_audio()` |
| **Async-safe audio streaming** with `sounddevice` | ✅ `run_coroutine_threadsafe` using `self.loop` |
| **Play audio via callback stream**              | ✅ `play_audio()` with `OutputStream` |
| **Handles `TokenTracker` if available**         | ✅ Conditional import and usage |
| **Model version switch via `model_type`**       | ✅ `model_map` with validation |
| **Prompt versioning with LangChain/LangSmith**  | ✅ Dynamic prompt loading via `LLangC_Prompt_Manager` |

---

## 📝 Prompt Versioning with LangChain/LangSmith

GeminiAudioAI loads its system prompt dynamically from LangChain/LangSmith, letting you manage and version prompts centrally—without hardcoding them into your codebase.

### How It Works

- **Prompt Manager:** Uses `LLangC_Prompt_Manager` to load the correct system prompt version at startup.
- **Prompt Version Flag:** Use the `--prompt_version N` argument when launching the app to choose the desired prompt:
    ```bash
    python main.py --mode screen --model_type 1 --prompt_version 1
    python main.py --mode screen --model_type 1 --prompt_version 2
    ```
- **Prompt Default:** Defaults to the latest prompt version.
- **Prompt Identifiers:** Each prompt version maps to a unique name (and optionally commit hash) in LangSmith.

#### How to Update Prompts

1. Go to your LangSmith dashboard.
2. Create or update your prompt in the repo (e.g., `geminaudioai_prompt_v2`).
3. Deploy a new commit if you want a specific version.
4. Use the correct version number (and optionally commit hash) in your app launch or config.

#### Tips for Prompt Management

1. Increment the prompt version number for major changes.
2. Store sample prompts and commit hashes in your documentation or `.env` for reference.
3. If you encounter a 404 or not found error, check that the prompt name and commit hash exist and are public in your LangSmith account.

#### Troubleshooting

- **404/400 Errors:** Double-check the prompt name, commit hash, and your LangSmith API key.
- **Prompt Not Updating:** Restart the app, or force reload the prompt by clearing any local caches or `.env` references.

---

## 📦 Model Types

| Type | Model Path (as of July 2024)                             | Use Case/Notes                   |
|------|----------------------------------------------------------|----------------------------------|
| 1    | models/gemini-2.5-flash-exp-native-audio-thinking-dialog | "Thinking" model, safest default |
| 2    | models/gemini-2.5-flash-preview-native-audio-dialog      | Function calls enabled           |
| 3    | models/gemini-live-2.5-flash-preview                     | All features, may be slower      |

- **Model Type Default:** Defaults to the first model (type - 1) version, with thinking capabilities.

---

## 🧪 Testing

All unit/integration tests are under `main/tests/`.

Use `pytest` or `unittest` to run tests:

```bash
pytest main/tests/
# or
python -m unittest discover main/tests
```

---

## 🔧 Development

- Update requirements as you add dependencies.
- Use Poetry or pip as you prefer for dependency management.
- Edit `setup.py` if packaging as a library.

---

## 📝 Troubleshooting

- **API 404/400:** Double-check API keys and that prompt names/versions match your LangSmith repo.
- **Prompt Not Updating:** Restart the app, check prompt name/version, or clear any local cache.
- **Audio/Video Errors:** Check your device config in `main/config.py`, and your microphone/camera permissions.

---