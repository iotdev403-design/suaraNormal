import torch
import torchaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from groq import Groq
from gtts import gTTS
import os
from dotenv import load_dotenv
import uuid
import io

# Change: Import Form to receive form fields
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# --- START OF MODEL CONFIGURATION ---
# We only need to define the local model here now
lora_config = {
    "id": "lora",
    "base_model": "openai/whisper-medium",
    "lora_path": "LoRA4"
}

# The Groq model name for transcription
groq_whisper_model = "whisper-large-v3"

# This dictionary will hold our locally loaded model and processor
loaded_models = {}

# --- NEW: Dictionary for multiple system prompts ---
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
# --- END OF NEW ---

# --- END OF MODEL CONFIGURATION ---
# --- Load Local Model at Startup ---

print("🚀 Initializing local models...")

# Load the fine-tuned LoRA model
try:
    print(f"📦 Loading LoRA model (base: {lora_config['base_model']})...")
    processor_lora = WhisperProcessor.from_pretrained(lora_config['base_model'])
    model_lora = WhisperForConditionalGeneration.from_pretrained(lora_config['base_model'])
    
    print(f"🔗 Attaching LoRA adapter from {lora_config['lora_path']}...")
    model_lora_peft = PeftModel.from_pretrained(model_lora, lora_config['lora_path'])
    
    # This correctly sets the language for the local model
    loaded_models[lora_config['id']] = {
        "processor": processor_lora,
        "model": model_lora_peft,
        "forced_decoder_ids": processor_lora.get_decoder_prompt_ids(language="indonesian", task="transcribe")
    }
    print("✅ LoRA model loaded successfully.")

except Exception as e:
    print(f"❌ Failed to load LoRA model: {e}")
    # exit()

# --- Groq Client Setup ---

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ GROQ_API_KEY not found in .env file. Exiting.")
    exit()
groq_client = Groq(api_key=groq_api_key)
groq_chat_model = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- FastAPI App ---

app = FastAPI()

# --- CORS CONFIGURATION ---
# MODIFIED: Changed origins and methods to ["*"] for easier development
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Audio Config ---

TARGET_RATE = 16000
latest_audio_path = None

# --- NEW: Function for hard-coded randomization ---
def randomize_text(transcription: str, num_words: int = 5) -> dict:
    """
    Selects a specified number of random words from the transcription.
    """
    words = transcription.split()
    if not words:
        return {"natural_text": "Transkrip kosong, tidak ada kata untuk diacak."}
    
    # If there are fewer words than requested, use all of them
    if len(words) < num_words:
        random_words = words
    else:
        random_words = np.random.choice(words, num_words, replace=False)
    
    natural_text = " ".join(random_words)
    return {"natural_text": natural_text}

# --- MODIFIED: Function now accepts a prompt selection ---
def translate_to_natural_sound_with_groq(transcription, prompt_selection):
    # Check if the selected prompt exists, otherwise default to the first one
    if prompt_selection not in SYSTEM_PROMPTS:
        print(f"⚠️ Warning: Invalid prompt selection '{prompt_selection}'. Defaulting to 'summarizer'.")
        prompt_selection = "summarizer"

    # Get the selected prompt and format it with the transcription
    base_prompt = SYSTEM_PROMPTS[prompt_selection]
    system_prompt = f"{base_prompt}\n\nTeks transkrip pengguna: \"{transcription}\""
    
    print(f"🧠 Using prompt key: '{prompt_selection}'")

    try:
        completion = groq_client.chat.completions.create(
            model=groq_chat_model, messages=[{"role": "system", "content": system_prompt}],
            temperature=0.0, max_tokens=100, top_p=1, stream=False
        )
        response_text = completion.choices[0].message.content.strip()
        import json
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"natural_text": response_text}
    except Exception as e:
        print(f"❌ Error with Groq API: {e}")
        return {"natural_text": "I am sorry, I could not process the sound."}

def speak_text_to_file(text, lang='id'):
    # This function is unchanged
    global latest_audio_path
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        os.makedirs("responses", exist_ok=True)
        speech_file = os.path.join("responses", filename)
        tts.save(speech_file)
        latest_audio_path = speech_file
        return speech_file
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")
        return None

# --- MODIFIED: Endpoint now accepts 'prompt_selection' from the form ---
@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    model_selection: str = Form(...),
    prompt_selection: str = Form(...) # <-- ADDED THIS LINE
):
    try:
        transcription = ""
        print(f"Processing with selection: {model_selection}")
        
        contents = await audio_file.read()

        if model_selection == lora_config['id']:
            # This path already correctly uses Indonesian settings from the loaded model
            if lora_config['id'] not in loaded_models:
                 return JSONResponse(status_code=500, content={"message": "Local LoRA model is not available."})
            
            print("🎤 Transcribing with local LoRA model (language: Indonesian)...")
            selected_assets = loaded_models[lora_config['id']]
            processor = selected_assets["processor"]
            model = selected_assets["model"]
            forced_decoder_ids = selected_assets["forced_decoder_ids"]

            audio_segment = AudioSegment.from_file(io.BytesIO(contents))
            audio_segment = audio_segment.set_frame_rate(TARGET_RATE).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            samples /= (1 << (8 * audio_segment.sample_width - 1))

            input_features = processor(samples, sampling_rate=TARGET_RATE, return_tensors="pt").input_features

            with torch.no_grad():
                predicted_ids = model.base_model.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        elif model_selection == 'large_v3':
            print("🎤 Transcribing with Groq API (language: Indonesian)...")
            
            transcription_response = groq_client.audio.transcriptions.create(
                file=(audio_file.filename, contents),
                model=groq_whisper_model,
                language="id"
            )
            transcription = transcription_response.text

        else:
            return JSONResponse(status_code=400, content={"message": "Invalid model selected."})
        
        print(f"Initial transcription: {transcription}")
        
        natural_text_dict = {}
        if prompt_selection == 'randomizer':
            print("🎲 Randomizing text with numpy...")
            natural_text_dict = randomize_text(transcription)
        else:
            # Pass the prompt selection to the processing function
            natural_text_dict = translate_to_natural_sound_with_groq(transcription, prompt_selection) # <-- MODIFIED THIS LINE

        natural_text = natural_text_dict.get("natural_text", "Could not process text.")
        speak_text_to_file(natural_text, lang='id')
        
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
    # This function is unchanged
    if latest_audio_path and os.path.exists(latest_audio_path):
        return FileResponse(latest_audio_path, media_type="audio/mpeg", filename="response.mp3")
    return JSONResponse(status_code=404, content={"message": "Audio file not found."})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)