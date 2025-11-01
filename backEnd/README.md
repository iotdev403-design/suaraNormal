# Audio Processing and Natural Language Response API

This project provides a FastAPI-based API for audio processing, transcription, and natural language response generation. It leverages both local and cloud-based AI models to offer flexibility and power in transforming spoken audio into summarized or randomized text, and then converting that text back into speech.

## 🌟 Highlights

*   **Dual Transcription Models**: Choose between a fine-tuned local LoRA Whisper model for Indonesian or the powerful `whisper-large-v3` model via the Groq API.
*   **Intelligent Text Processing**: Utilizes Groq's Llama 4 model to generate concise, natural language summaries of transcribed text.
*   **Text Randomization**: A feature to select a random subset of words from the transcription.
*   **Text-to-Speech**: Converts the processed text back into an audible MP3 format using gTTS.
*   **Flexible API**: Built with FastAPI, providing a robust and easy-to-use interface for audio processing tasks.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   [Conda](https://docs.conda.io/en/latest/miniconda.html) package and environment manager.
*   Access to the Groq API and a corresponding API key.
*   Git for cloning the repository.

### ⬇️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/stefanus-ai-tech/peloSTTgitNew
    cd peloSTTgitNew
    ```

2.  **Create a `requirements.txt` file:**
    In the root directory of the project, create a file named `requirements.txt` and paste the following dependencies into it:

    ```
    fastapi
    uvicorn[standard]
    python-dotenv
    groq
    gTTS
    pydub
    numpy
    peft
    python-multipart
    transformers
    torch
    torchaudio
    ```

3.  **Create and activate the Conda environment:**
    ```bash
    # Create a new environment named 'audio_api' with Python 3.9
    conda create --name audio_api python=3.9 -y

    # Activate the new environment
    conda activate audio_api
    ```

4.  **Install dependencies:**
    With your conda environment active, install all the required packages using pip and the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```
    > **Note on PyTorch**: For systems with NVIDIA GPUs, you might achieve better performance by installing PyTorch using conda's specific command. You can find the correct command for your system on the [PyTorch website](https://pytorch.org/get-started/locally/).

5.  **Set up your environment variables:**
    Create a `.env` file in the root directory of the project and add your Groq API key:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

6.  **Download the local model:**
    Ensure the `LoRA4` model files are present in the specified path within the project. The configuration in the code points to a directory named `LoRA4`.

### 🏃‍♀️ Running the Application

Make sure your `audio_api` conda environment is active. Then, to start the FastAPI server, run the following command in your terminal:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

The API will then be accessible at `http://localhost:8001`.

## 🤖 Usage

The primary functionality is exposed through a single endpoint that processes audio files.

### `/process_audio`

*   **Method:** `POST`
*   **Description:** Transcribes an audio file using the selected model, processes the transcription based on the chosen prompt, and generates an audio response.
*   **Form Data:**
    *   `audio_file`: The audio file to be processed (e.g., in `.wav`, `.mp3` format).
    *   `model_selection`: A string indicating the transcription model to use.
        *   `lora`: For the local fine-tuned Whisper model.
        *   `large_v3`: For the Groq `whisper-large-v3` model.
    *   `prompt_selection`: A string to determine how the transcribed text is processed.
        *   `summarizer`: To generate a concise summary.
        *   `randomizer`: To get a random selection of words from the transcript.
*   **Responses:**
    *   `200 OK`: Returns a JSON object with the initial transcription and the processed natural language text.
    *   `400 Bad Request`: If an invalid model is selected.
    *   `500 Internal Server Error`: For any other issues during processing.

### `/get_response_audio`

*   **Method:** `GET`
*   **Description:** Retrieves the latest generated audio response as an MP3 file.
*   **Responses:**
    *   `200 OK`: Returns the audio file in `audio/mpeg` format.
    *   `404 Not Found`: If no audio file has been generated yet.

## 🛠️ Configuration

The application's behavior can be customized by modifying the following dictionaries in the source code:

*   **`lora_config`**: Defines the configuration for the local LoRA model, including its ID, base model, and the path to the LoRA adapter.
*   **`SYSTEM_PROMPTS`**: A dictionary where you can add or modify system prompts for the Groq Llama model. The keys in this dictionary are used as the `prompt_selection` in the API call. This allows for easy extension with new text processing functionalities.

## ⚙️ Technologies Used

*   **[FastAPI](https://fastapi.tiangolo.com/)**: A modern, fast (high-performance) web framework for building APIs with Python.
*   **[PyTorch](https://pytorch.org/)**: An open-source machine learning library.
*   **[Hugging Face Transformers](https://huggingface.co/transformers/)**: Provides the Whisper model for audio transcription.
*   **[PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index)**: For loading the LoRA fine-tuned model.
*   **[Groq](https://groq.com/)**: Powers the high-speed `whisper-large-v3` transcription and the Llama 4 for natural language processing.
*   **[gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/en/latest/)**: For converting text into speech.
*   **[Pydub](https://github.com/jiaaro/pydub)**: For audio manipulation.
*   **[Uvicorn](https://www.uvicorn.org/)**: An ASGI server for running the FastAPI application.
