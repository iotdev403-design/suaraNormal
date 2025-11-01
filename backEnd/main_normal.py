import io
import os
import uuid
import json
import traceback
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS

# --- NEW IMPORTS for Personalized Voice ---
from TTS.api import TTS
from speechbrain.pretrained import SpeakerRecognition
# torchaudio is a dependency for speechbrain, good to have it explicit
import torchaudio 

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
groq_chat_model = "meta-llama/llama-4-scout-17b-16e-instruct" 

# --- System Prompts for Different Tasks ---
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

# --- NEW: Speaker Verification & Personalized TTS Setup ---

# This flag will be updated by the speaker verification logic
isMe = False

try:
    # --- Speaker Verification Setup ---
    print("🔊 Loading Speaker Verification model...")
    speaker_verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("✅ Speaker Verification model loaded.")

    # This is the reference audio of your voice.
    # IMPORTANT: Record yourself saying a sentence and save it as "my_voice_reference.wav" in the same directory.
    MY_VOICE_REFERENCE = "my_voice_reference.wav"
    if not os.path.exists(MY_VOICE_REFERENCE):
        print(f"❌ WARNING: Voice reference file not found at '{MY_VOICE_REFERENCE}'. Speaker verification will fail.")


    # --- Personalized TTS Model Setup ---
    # IMPORTANT: These paths assume you have run the training script and the model exists in 'my_trained_model/'.
    personalized_model_config = "my_trained_model/config.json"
    personalized_model_file = "my_trained_model/best_model.pth"
    
    if os.path.exists(personalized_model_config) and os.path.exists(personalized_model_file):
        print("🔊 Loading Personalized TTS model...")
        personalized_tts = TTS(
            model_path=personalized_model_file,
            config_path=personalized_model_config,
            progress_bar=False,
            gpu=False # Set to True if you have a GPU and compatible PyTorch
        )
        print("✅ Personalized TTS model loaded.")
    else:
        personalized_tts = None
        print("❌ WARNING: Personalized TTS model not found. The 'personalized' option will not work.")

except Exception as e:
    print(f"❌ Critical error during model loading: {e}")
    print("   The application might not function correctly for personalized voice features.")
    speaker_verifier = None
    personalized_tts = None


# --- END OF API CONFIGURATION ---

# --- FastAPI App ---
app = FastAPI()

# --- CORS CONFIGURATION ---
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

# --- NEW: Speaker Verification Function ---
def verify_speaker(audio_file_path: str):
    """
    Verifies if the speaker in the audio file matches the reference voice.
    """
    global isMe
    if not speaker_verifier or not os.path.exists(MY_VOICE_REFERENCE):
        print("❌ Cannot perform speaker verification. Model or reference file is missing.")
        isMe = False
        return False
    try:
        score_threshold = 0.5 
        score, prediction = speaker_verifier.verify_files(MY_VOICE_REFERENCE, audio_file_path)
        print(f"🎤 Speaker verification score: {score[0]:.2f} (Threshold: {score_threshold})")
        if prediction[0]:
            print("✅ Speaker VERIFIED.")
            isMe = True
            return True
        else:
            print("❌ Speaker REJECTED.")
            isMe = False
            return False
    except Exception as e:
        print(f"❌ Error during speaker verification: {e}")
        isMe = False
        return False

# --- NEW: Personalized Speech Synthesis Function ---
def speak_text_to_file_personalized(text: str):
    """
    Converts text to speech using your trained personalized voice model.
    """
    if not personalized_tts:
        print("❌ Cannot generate personalized speech. The personalized TTS model is not loaded.")
        return None
    try:
        if not text or not text.strip():
            print("⚠️ Warning: Attempted to generate personalized speech from empty text.")
            return None
        filename = f"response_{uuid.uuid4()}.wav" # Coqui TTS outputs wav
        os.makedirs("responses", exist_ok=True)
        speech_file = os.path.join("responses", filename)
        personalized_tts.tts_to_file(text=text, file_path=speech_file)
        return speech_file
    except Exception as e:
        print(f"❌ Failed to generate personalized speech: {e}")
        return None

# --- Existing Functions (Unchanged) ---
def translate_to_natural_sound_with_groq(transcription: str, prompt_selection: str):
    if prompt_selection not in SYSTEM_PROMPTS:
        print(f"⚠️ Warning: Invalid prompt selection '{prompt_selection}'. Defaulting to 'summarizer'.")
        prompt_selection = "summarizer"
    base_prompt = SYSTEM_PROMPTS[prompt_selection]
    system_prompt = f"{base_prompt}\n\nTeks transkrip pengguna: \"{transcription}\""
    print(f"🧠 Using prompt key: '{prompt_selection}'")
    try:
        completion = groq_client.chat.completions.create(
            model=groq_chat_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        response_text = completion.choices[0].message.content.strip()
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Groq response was not valid JSON. Response: {response_text}")
            return {"natural_text": response_text}
    except Exception as e:
        print(f"❌ Error with Groq API: {e}")
        return {"natural_text": "I am sorry, I could not process the sound."}

def speak_text_to_file(text: str, lang: str = 'id'):
    try:
        if not text or not text.strip():
            print("⚠️ Warning: Attempted to generate speech from empty text.")
            return None
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        os.makedirs("responses", exist_ok=True)
        speech_file = os.path.join("responses", filename)
        tts.save(speech_file)
        return speech_file
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")
        return None

# --- API Endpoints ---

# --- UPDATED /process_audio Endpoint ---
@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    model_selection: str = Form(...),
    prompt_selection: str = Form(...)
):
    global latest_summary_audio_path, latest_transcription_audio_path

    # Save the uploaded file temporarily for verification and transcription
    temp_audio_path = f"temp_{uuid.uuid4()}.webm"
    try:
        contents = await audio_file.read()
        with open(temp_audio_path, "wb") as f:
            f.write(contents)

        # --- SPEAKER VERIFICATION LOGIC ---
        if model_selection == "personalized":
            print("🕵️ 'Personalized' model selected. Running speaker verification...")
            if not verify_speaker(temp_audio_path):
                return JSONResponse(
                    status_code=403, # Forbidden
                    content={"message": "Speaker verification failed. You are not authorized to use this voice."}
                )

        # --- Transcription ---
        print("🎤 Transcribing with Groq API (language: Indonesian)...")
        transcription_response = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, contents), model=groq_whisper_model, language="id"
        )
        transcription = transcription_response.text
        print(f"Initial transcription: {transcription}")
        
        # --- Text Processing ---
        natural_text_dict = translate_to_natural_sound_with_groq(transcription, prompt_selection)
        natural_text = natural_text_dict.get("natural_text", "Could not process text.")
        
        # --- DYNAMIC SPEECH SYNTHESIS ---
        if model_selection == "personalized" and isMe:
            print("🔊 Using PERSONALIZED voice for summary...")
            latest_summary_audio_path = speak_text_to_file_personalized(natural_text)
        else:
            print("🔊 Using NORMAL voice (gTTS) for summary...")
            latest_summary_audio_path = speak_text_to_file(natural_text, lang='id')
        
        # Original transcription always uses the standard voice
        latest_transcription_audio_path = speak_text_to_file(transcription, lang='id')
        
        return JSONResponse(content={
            "initial_transcription": transcription,
            "natural_text": natural_text,
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.get("/get_response_audio")
async def get_response_audio():
    if latest_summary_audio_path and os.path.exists(latest_summary_audio_path):
        # Determine media type based on file extension
        media_type = "audio/wav" if latest_summary_audio_path.endswith(".wav") else "audio/mpeg"
        return FileResponse(latest_summary_audio_path, media_type=media_type, filename="response.mp3")
    return JSONResponse(status_code=404, content={"message": "Audio file not found."})


@app.get("/get_transcription_audio")
async def get_transcription_audio():
    if latest_transcription_audio_path and os.path.exists(latest_transcription_audio_path):
        return FileResponse(latest_transcription_audio_path, media_type="audio/mpeg", filename="transcription.mp3")
    return JSONResponse(status_code=404, content={"message": "Transcription audio file not found."})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)