

import gtts from 'gtts';
import { promisify } from 'util';
import 'dotenv/config'; // Load environment variables from .env file
import fs from 'fs';
import path from 'path';
import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { Groq } from 'groq-sdk';
import googleTTS from 'google-tts-api';
import { randomUUID } from 'crypto'; // For unique filenames
import { AutoProcessor, WavLMForXVector } from '@xenova/transformers';
import ffmpeg from 'fluent-ffmpeg';
import wav from 'node-wav';




// --- Configuration ---
const app = express();
const PORT = 8001;
const UPLOADS_DIR = 'uploads';
const RESPONSES_DIR = 'responses';
const ENROLLMENT_FILE = 'audio_saya.mp3'; // Our reference voice file
const THRESHOLD = 0.69; // Speaker verification threshold

// --- Groq Client Setup ---
const groqApiKey = process.env.GROQ_API_KEY;
if (!groqApiKey) {
    console.error("❌ GROQ_API_KEY tidak ditemukan di file .env. Program akan berhenti.");
    process.exit(1);
}
const groq = new Groq({ apiKey: groqApiKey });

// --- System Prompts for Different Tasks ---
const SYSTEM_PROMPTS = {
    "summarizer": `Kamu adalah summarizer ekstrim teks transkrip bahasa Indonesia. Hanya intinya saja, jangan menuliskan ulang semuanya, maksimal 5 kata. Tugasmu adalah membaca teks transkrip lalu menghasilkan SATU kalimat ringkas yang alami dan mewakili maksud utama dari transkrip tersebut. Gunakan kata-kata yang wajar digunakan sehari-hari. Jangan memberi penjelasan, variasi, atau alternatif. Jangan menambahkan tanda baca kecuali tanda baca normal yang memang diperlukan. Jawab HANYA dalam format JSON persis seperti ini: {"natural_text": "<hasil kamu>"} Tanpa teks tambahan, tanpa catatan, dan tanpa field lain.`
};

// --- Middleware & Directory Setup ---
app.use(cors());
app.use(express.json());

// Create required directories if they don't exist
[UPLOADS_DIR, RESPONSES_DIR].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir);
});

// Multer setup for temporary file uploads, preserving file extension
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, UPLOADS_DIR);
    },
    filename: function (req, file, cb) {
        // Membuat nama file unik sambil mempertahankan ekstensi aslinya
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const extension = path.extname(file.originalname);
        cb(null, file.fieldname + '-' + uniqueSuffix + extension);
    }
});
const upload = multer({ storage: storage });

// --- Global variables to store latest audio paths (similar to Python version) ---
let latest_summary_audio_path = null;
let latest_transcription_audio_path = null;

// --- Model Loading (Speaker Verification Model) ---
let verifierProcessor, verifierModel;
async function loadVerificationModel() {
    console.log("🔊 Memuat model Speaker Verification...");
    try {
        verifierProcessor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
        verifierModel = await WavLMForXVector.from_pretrained('Xenova/wavlm-base-plus-sv');
        console.log("✅ Model Speaker Verification berhasil dimuat.");
    } catch (error) {
        console.error("❌ Gagal memuat model Speaker Verification:", error);
        process.exit(1);
    }
}

// --- Helper Functions (Audio, AI, and TTS) ---

// Speaker Verification Function (Adapted from previous Node.js script)
async function verifySpeaker(verificationAudioPath) {
    if (!verifierModel) {
        console.error("❌ Model verifikasi belum dimuat.");
        return false;
    }
    console.log("🕵️  Menjalankan verifikasi suara...");

    const enrollmentWav = path.join(UPLOADS_DIR, 'enrollment.wav');
    const verificationWav = path.join(UPLOADS_DIR, `verify_${randomUUID()}.wav`);

    try {
        // Convert both files to WAV format required by the model
        await Promise.all([
            new Promise((res, rej) => ffmpeg(ENROLLMENT_FILE).audioFrequency(16000).audioChannels(1).toFormat('wav').on('end', res).on('error', rej).save(enrollmentWav)),
            new Promise((res, rej) => ffmpeg(verificationAudioPath).audioFrequency(16000).audioChannels(1).toFormat('wav').on('end', res).on('error', rej).save(verificationWav))
        ]);

        // Function to extract embedding
        const extractEmbedding = async (audioPath) => {
            const buffer = fs.readFileSync(audioPath);
            const { channelData } = wav.decode(buffer);
            const inputs = await verifierProcessor(channelData[0], { sampling_rate: 16000 });
            const { embeddings } = await verifierModel(inputs);
            return Array.from(embeddings.data);
        };
        
        // Get embeddings for both audio files
        const [enrollmentEmbed, verificationEmbed] = await Promise.all([
            extractEmbedding(enrollmentWav),
            extractEmbedding(verificationWav)
        ]);

        // Calculate Cosine Similarity
        const cosineSimilarity = (vecA, vecB) => {
            const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
            const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
            const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
            return dotProduct / (magnitudeA * magnitudeB);
        };

        const similarity = cosineSimilarity(enrollmentEmbed, verificationEmbed);
        console.log(`🎤 Skor verifikasi pembicara: ${similarity.toFixed(4)} (Ambang Batas: ${THRESHOLD})`);
        
        return similarity > THRESHOLD;
    } finally {
        // Clean up temporary WAV files
        fs.unlink(enrollmentWav, () => {});
        fs.unlink(verificationWav, () => {});
    }
}

// Groq Text Processing Function
async function summarizeWithGroq(transcription, promptKey = 'summarizer') {
    const systemPrompt = `${SYSTEM_PROMPTS[promptKey]}\n\nTeks transkrip pengguna: "${transcription}"`;
    try {
        const chatCompletion = await groq.chat.completions.create({
            messages: [{ role: 'system', content: systemPrompt }],
            model: 'meta-llama/llama-4-scout-17b-16e-instruct', // Or another model like 'meta-llama/llama-4-scout-17b-16e-instruct'
            temperature: 0.0,
            max_tokens: 100,
        });
        const responseText = chatCompletion.choices[0]?.message?.content || '';
        return JSON.parse(responseText);
    } catch (error) {
        console.error("❌ Error dengan Groq Chat API:", error);
        return { natural_text: "Maaf, saya tidak bisa memproses suara itu." };
    }
}





// Standard Text-to-Speech Function menggunakan gtts
async function speakWithGTTS(text, lang = 'id') {
    if (!text || !text.trim()) {
        console.warn("⚠️ Mencoba menghasilkan audio dari teks kosong.");
        return null;
    }
    try {
        const filePath = path.join(RESPONSES_DIR, `response_${randomUUID()}.mp3`);
        
        const speech = new gtts(text, lang);
        
        // Konversi callback ke promise
        await new Promise((resolve, reject) => {
            speech.save(filePath, (err) => {
                if (err) reject(err);
                else resolve();
            });
        });
        
        console.log(`✅ Audio berhasil dibuat: ${filePath}`);
        return filePath;
    } catch (error) {
        console.error("❌ Gagal menghasilkan audio (gTTS):", error);
        console.error("Error details:", error.message);
        return null;
    }
}

// Personalized Text-to-Speech Function (Placeholder)
async function speakWithPersonalizedTTS(text) {
    console.log("⚠️ FUNGSI PLACEHOLDER: Menghasilkan audio dengan suara personal.");
    // IN A REAL SCENARIO: You would call your local Coqui-TTS or similar model here.
    // For now, we will just use the standard voice as a fallback.
    return speakWithGTTS(text, 'id'); // Fallback to standard voice
}


// --- API Endpoints ---

app.post('/process_audio', upload.single('audio_file'), async (req, res) => {
    console.log("\n🚀 Permintaan diterima di /process_audio");

    if (!req.file) {
        return res.status(400).json({ message: 'Tidak ada file audio yang diunggah.' });
    }

    const { model_selection, prompt_selection } = req.body;
    const tempAudioPath = req.file.path;
    let isMe = false; // Flag to track verification status

    try {
        // --- 1. Speaker Verification (if 'personalized' is selected) ---
        if (model_selection === 'personalized') {
            const verificationResult = await verifySpeaker(tempAudioPath);
            if (!verificationResult) {
                console.log("❌ Verifikasi GAGAL. Pengguna tidak diizinkan.");
                return res.status(403).json({ message: "VERIFIKASI GAGAL, PEMBICARA TIDAK DIKENAL" });
            }
            console.log("✅ Verifikasi BERHASIL.");
            isMe = true;
        }

        // --- 2. Transcription with Groq ---
        console.log("🎤 Mentranskripsi dengan Groq API...");
        const transcription = await groq.audio.transcriptions.create({
            file: fs.createReadStream(tempAudioPath),
            model: "whisper-large-v3",
            language: "id"
        });
        const initialTranscription = transcription.text;
        console.log(`   Transkripsi awal: ${initialTranscription}`);

        // --- 3. Text Processing (Summarization) ---
        console.log("🧠 Memproses teks dengan Groq Chat...");
        const naturalTextDict = await summarizeWithGroq(initialTranscription, prompt_selection);
        const naturalText = naturalTextDict.natural_text;
        console.log(`   Teks ringkasan: ${naturalText}`);

        // --- 4. Dynamic Speech Synthesis ---
        if (model_selection === 'personalized' && isMe) {
            console.log("🔊 Menggunakan suara PERSONALIZED untuk ringkasan...");
            latest_summary_audio_path = await speakWithPersonalizedTTS(naturalText);
        } else {
            console.log("🔊 Menggunakan suara NORMAL (Google) untuk ringkasan...");
            latest_summary_audio_path = await speakWithGTTS(naturalText, 'id');
        }

        // Original transcription always uses the standard voice
        latest_transcription_audio_path = await speakWithGTTS(initialTranscription, 'id');

        // --- 5. Send Response ---
        return res.status(200).json({
            initial_transcription: initialTranscription,
            natural_text: naturalText,
        });

    } catch (error) {
        console.error("❌ Terjadi error pada /process_audio:", error);
        return res.status(500).json({ message: `Terjadi error di server: ${error.message}` });
    } finally {
        // Clean up the uploaded temporary file
        fs.unlink(tempAudioPath, (err) => {
            if (err) console.error("Gagal menghapus file sementara:", err);
        });
    }
});

app.get('/get_response_audio', (req, res) => {
    if (latest_summary_audio_path && fs.existsSync(latest_summary_audio_path)) {
        res.sendFile(path.resolve(latest_summary_audio_path));
    } else {
        res.status(404).json({ message: "File audio tidak ditemukan." });
    }
});

app.get('/get_transcription_audio', (req, res) => {
    if (latest_transcription_audio_path && fs.existsSync(latest_transcription_audio_path)) {
        res.sendFile(path.resolve(latest_transcription_audio_path));
    } else {
        res.status(404).json({ message: "File audio transkripsi tidak ditemukan." });
    }
});


// --- Start Server ---
app.listen(PORT, () => {
    console.log(`🚀 Server berjalan di http://localhost:${PORT}`);
    // Load the AI model after the server is ready
    loadVerificationModel();
});
