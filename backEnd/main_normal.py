import io
import os
import uuid
import json
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS

# --- START OF API CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# The Groq model name for transcription
groq_whisper_model = "whisper-large-v3"

# --- Groq Client Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ GROQ_API_KEY not found in .env file. Exiting.")
    exit()
groq_client = Groq(api_key=groq_api_key)
groq_chat_model = "meta-llama/llama-4-scout-17b-16e-instruct" # Using a standard, available model

# --- System Prompts for Different Tasks ---
# The keys (e.g., "summarizer") will be sent from the frontend.
SYSTEM_PROMPTS = {
    "summarizer": (
        "Kamu adalah summarizer ekstrim teks transkrip bahasa Indonesia. Hanya intinya saja, jangan menuliskan ulang semuanya, maksimal 5 kata. "
        "Tugasmu adalah membaca teks transkrip lalu menghasilkan SATU kalimat ringkas yang alami dan mewakili maksud utama dari transkrip tersebut. "
        "Gunakan kata-kata yang wajar digunakan sehari-hari. "
        "Jangan memberi penjelasan, variasi, atau alternatif. "
        "Jangan menambahkan tanda baca kecuali tanda baca normal yang memang diperlukan. "
        "Jawab HANYA dalam format JSON persis seperti ini: {\"natural_text\": \"<hasil kamu>\"} "
        "Tanpa teks tambahan, tanpa catatan, dan tanpa field lain."
    )
}

# --- END OF API CONFIGURATION ---

# --- FastAPI App ---

app = FastAPI()

# --- CORS CONFIGURATION ---
# Allows all origins for easier development
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Audio Config ---
latest_summary_audio_path = None
latest_transcription_audio_path = None

# --- Text Processing and Speech Synthesis Functions ---

def translate_to_natural_sound_with_groq(transcription: str, prompt_selection: str):
    """
    Uses Groq's chat completion to process the transcription based on a selected prompt.
    """
    # Default to the 'summarizer' prompt if the selection is invalid
    if prompt_selection not in SYSTEM_PROMPTS:
        print(f"⚠️ Warning: Invalid prompt selection '{prompt_selection}'. Defaulting to 'summarizer'.")
        prompt_selection = "summarizer"

    # Get the selected prompt and format it
    base_prompt = SYSTEM_PROMPTS[prompt_selection]
    system_prompt = f"{base_prompt}\n\nTeks transkrip pengguna: \"{transcription}\""
    
    print(f"🧠 Using prompt key: '{prompt_selection}'")

    try:
        completion = groq_client.chat.completions.create(
            model=groq_chat_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.0,
            max_tokens=100,
            top_p=1,
            stream=False
        )
        response_text = completion.choices[0].message.content.strip()
        
        # Try to parse the response as JSON, otherwise return the raw text
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Groq response was not valid JSON. Returning raw text. Response: {response_text}")
            return {"natural_text": response_text}
            
    except Exception as e:
        print(f"❌ Error with Groq API: {e}")
        return {"natural_text": "I am sorry, I could not process the sound."}

def speak_text_to_file(text: str, lang: str = 'id'):
    """
    Converts text to speech using gTTS and saves it as an MP3 file.
    Returns the path to the created file.
    """
    try:
        # Menambahkan pencegahan error jika teks kosong
        if not text or not text.strip():
            print("⚠️ Warning: Attempted to generate speech from empty text.")
            return None
            
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        os.makedirs("responses", exist_ok=True)
        speech_file = os.path.join("responses", filename)
        tts.save(speech_file)
        # Fungsi ini sekarang hanya mengembalikan path, tidak mengubah variabel global
        return speech_file
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")
        return None

# --- API Endpoints ---

@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    prompt_selection: str = Form(...)
):
    """
    Receives an audio file, transcribes it, processes the transcription, 
    and generates spoken responses for BOTH the original and processed text.
    """
    global latest_summary_audio_path, latest_transcription_audio_path # Panggil kedua variabel global

    try:
        contents = await audio_file.read()
        
        print("🎤 Transcribing with Groq API (language: Indonesian)...")
        audio_file_for_api = (audio_file.filename, contents)
        transcription_response = groq_client.audio.transcriptions.create(
            file=audio_file_for_api, model=groq_whisper_model, language="id"
        )
        transcription = transcription_response.text
        print(f"Initial transcription: {transcription}")
        
        natural_text_dict = translate_to_natural_sound_with_groq(transcription, prompt_selection)
        natural_text = natural_text_dict.get("natural_text", "Could not process text.")
        
        # --- PERUBAHAN UTAMA: Speech Synthesis untuk DUA file ---
        # 1. Buat audio untuk teks ringkasan (summary)
        latest_summary_audio_path = speak_text_to_file(natural_text, lang='id')
        
        # 2. Buat audio untuk teks transkripsi asli
        latest_transcription_audio_path = speak_text_to_file(transcription, lang='id')
        
        return JSONResponse(content={
            "initial_transcription": transcription,
            "natural_text": natural_text,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})

@app.get("/get_response_audio")
async def get_response_audio():
    """
    Serves the latest generated audio response file (for the summary).
    """
    if latest_summary_audio_path and os.path.exists(latest_summary_audio_path):
        return FileResponse(latest_summary_audio_path, media_type="audio/mpeg", filename="response.mp3")
    return JSONResponse(status_code=404, content={"message": "Audio file not found."})


@app.get("/get_transcription_audio")
async def get_transcription_audio():
    """
    Serves the latest generated audio file for the original transcription.
    """
    if latest_transcription_audio_path and os.path.exists(latest_transcription_audio_path):
        return FileResponse(latest_transcription_audio_path, media_type="audio/mpeg", filename="transcription.mp3")
    return JSONResponse(status_code=404, content={"message": "Transcription audio file not found."})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)