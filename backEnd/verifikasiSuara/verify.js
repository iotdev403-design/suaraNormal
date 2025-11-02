const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');

// For audio processing - SIMPLIFIED VERSION
let AutoProcessor, WavLMForXVector;
const ffmpeg = require('fluent-ffmpeg');
const wav = require('node-wav');

// --- Configuration ---
const ENROLLMENT_FILE = 'audio_saya.mp3';
const ENROLLMENT_WAV = 'enrollment.wav';
const VERIFICATION_WAV = 'verification.wav';
const SAMPLE_RATE = 16000;
const THRESHOLD = 0.70;

// Initialize readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Promisify readline question
const question = (query) => new Promise((resolve) => rl.question(query, resolve));

// --- Utility Functions ---
function printSeparator(char = '=', length = 60) {
  console.log(char.repeat(length));
}

function printProgress(progress, total) {
  const percentage = (progress / total) * 100;
  const bars = Math.floor(percentage / 5);
  const empty = 20 - bars;
  process.stdout.write(`\r   Progress: [${'█'.repeat(bars)}${'░'.repeat(empty)}] ${percentage.toFixed(0)}%`);
}

// --- Audio Processing Functions ---
async function getAudioDuration(filename) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filename, (err, metadata) => {
      if (err) reject(err);
      else resolve(metadata.format.duration);
    });
  });
}

async function convertAndTrimAudio(inputFile, outputFile, duration = null) {
  return new Promise((resolve, reject) => {
    let command = ffmpeg(inputFile)
      .audioFrequency(SAMPLE_RATE)
      .audioChannels(1)
      .format('wav');
    
    if (duration) {
      command = command.duration(duration);
    }
    
    command
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .save(outputFile);
  });
}

// Platform-specific recording command detection
function getRecordCommand() {
  const platform = process.platform;
  
  if (platform === 'win32') {
    return {
      command: 'ffmpeg',
      args: (duration, output) => [
        '-f', 'dshow',
        '-i', 'audio=Microphone',
        '-t', duration.toString(),
        '-ar', SAMPLE_RATE.toString(),
        '-ac', '1',
        '-y',
        output
      ]
    };
  } else if (platform === 'darwin') {
    return {
      command: 'ffmpeg',
      args: (duration, output) => [
        '-f', 'avfoundation',
        '-i', ':0',
        '-t', duration.toString(),
        '-ar', SAMPLE_RATE.toString(),
        '-ac', '1',
        '-y',
        output
      ]
    };
  } else {
    return {
      command: 'ffmpeg',
      args: (duration, output) => [
        '-f', 'pulse',
        '-i', 'default',
        '-t', duration.toString(),
        '-ar', SAMPLE_RATE.toString(),
        '-ac', '1',
        '-y',
        output
      ]
    };
  }
}

async function recordAudio(filename, duration = 10) {
  return new Promise((resolve, reject) => {
    console.log(`\n🔴 REKAMAN DIMULAI! (Durasi: ${duration} detik)`);
    console.log(`   Silakan berbicara sekarang...`);
    console.log(`   💡 Tip: Bacakan narasi yang sama untuk hasil terbaik`);
    
    const recordCmd = getRecordCommand();
    const args = recordCmd.args(duration, filename);
    
    const ffmpegProcess = spawn(recordCmd.command, args, {
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    let elapsed = 0;
    const interval = setInterval(() => {
      elapsed++;
      printProgress(elapsed, duration);
    }, 1000);
    
    let stderrData = '';
    ffmpegProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    ffmpegProcess.on('close', (code) => {
      clearInterval(interval);
      process.stdout.write('\n');
      
      if (code === 0) {
        console.log(`✅ Rekaman selesai!`);
        console.log(`   💾 Audio disimpan ke '${filename}'`);
        resolve();
      } else {
        reject(new Error(`Recording failed with code ${code}\n${stderrData}`));
      }
    });
    
    ffmpegProcess.on('error', (err) => {
      clearInterval(interval);
      process.stdout.write('\n');
      reject(err);
    });
  });
}

// --- Vector/Embedding Functions ---
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function loadAudioData(audioPath) {
  const buffer = fs.readFileSync(audioPath);
  const result = wav.decode(buffer);
  
  // Normalize to float32 array between -1 and 1
  const audioData = new Float32Array(result.channelData[0].length);
  for (let i = 0; i < result.channelData[0].length; i++) {
    audioData[i] = result.channelData[0][i];
  }
  
  return {
    data: audioData,
    sampleRate: result.sampleRate
  };
}

async function extractEmbedding(audioPath, processor, model) {
  // Load audio
  const audio = await loadAudioData(audioPath);
  
  // Process with the processor
  const inputs = await processor(audio.data, { sampling_rate: SAMPLE_RATE });
  
  // Get embeddings from model
  const { embeddings } = await model(inputs);
  
  // Convert to regular array
  return Array.from(embeddings.data);
}

// --- Main Program ---
async function main() {
  try {

    const transformers = await import('@xenova/transformers');
    ({ AutoProcessor, WavLMForXVector } = transformers);


    console.log("🎙️  SISTEM VERIFIKASI SUARA DENGAN TRANSFORMERS");
    printSeparator();
    
    // --- Step 1: Create Enrollment File ---
    console.log(`\n📂 Memuat audio enrollment dari '${ENROLLMENT_FILE}'...`);
    
    if (!fs.existsSync(ENROLLMENT_FILE)) {
      throw new Error(`File '${ENROLLMENT_FILE}' tidak ditemukan!`);
    }
    
    const totalDuration = await getAudioDuration(ENROLLMENT_FILE);
    console.log(`✅ Audio berhasil dimuat!`);
    console.log(`   Durasi: ${totalDuration.toFixed(2)} detik`);
    console.log(`   Sample Rate: ${SAMPLE_RATE} Hz`);
    
    if (totalDuration < 10) {
      console.log(`⚠️  PERINGATAN: Audio terlalu pendek (${totalDuration.toFixed(2)}s).`);
      console.log(`   Minimal 10 detik direkomendasikan untuk hasil optimal.`);
    }
    
    // Use first 12 seconds or entire audio
    const enrollmentDuration = Math.min(12, totalDuration);
    await convertAndTrimAudio(ENROLLMENT_FILE, ENROLLMENT_WAV, enrollmentDuration);
    
    console.log(`\n✅ File enrollment dibuat:`);
    console.log(`   📝 ${ENROLLMENT_WAV}: ${enrollmentDuration.toFixed(2)} detik`);
    
    // --- Step 2: Record Verification Audio ---
    printSeparator();
    console.log("🎤 TAHAP VERIFIKASI - REKAM SUARA ANDA");
    printSeparator();
    
    console.log("\nBerapa detik Anda ingin merekam? (default: 10 detik)");
    console.log("Tekan Enter untuk menggunakan default");
    
    const durasiInput = await question("Durasi (detik): ");
    let durasiRekam = parseInt(durasiInput) || 10;
    
    if (durasiRekam < 3) {
      console.log("⚠️  Durasi terlalu pendek. Minimal 3 detik. Menggunakan 5 detik.");
      durasiRekam = 5;
    } else if (durasiRekam > 30) {
      console.log("⚠️  Durasi terlalu panjang. Maksimal 30 detik.");
      durasiRekam = 30;
    }
    
    console.log(`\n📌 Bersiap untuk merekam selama ${durasiRekam} detik...`);
    await question("   Tekan ENTER untuk mulai merekam...");
    
    await recordAudio(VERIFICATION_WAV, durasiRekam);
    
    // --- Step 3: Load Model and Extract Embeddings ---
    console.log(`\n🤖 Memuat model speaker verification...`);
    console.log(`   Model: microsoft/wavlm-base-plus-sv`);
    
    // Using WavLM model specifically designed for speaker verification
    const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
    const model = await WavLMForXVector.from_pretrained('Xenova/wavlm-base-plus-sv');
    
    console.log("✅ Model berhasil dimuat!");
    
    console.log(`\n🔧 Memproses audio untuk verifikasi...`);
    console.log("✅ Audio berhasil diproses!");
    
    console.log(`\n🧬 Mengekstrak fitur suara (voice embeddings)...`);
    
    const enrollmentEmbed = await extractEmbedding(ENROLLMENT_WAV, processor, model);
    const verificationEmbed = await extractEmbedding(VERIFICATION_WAV, processor, model);
    
    console.log(`✅ Embedding berhasil diekstrak!`);
    console.log(`   Dimensi embedding: ${enrollmentEmbed.length}D`);
    
    // --- Step 4: Calculate Similarity ---
    console.log(`\n📊 Menghitung kesamaan suara...`);
    
    const similarity = cosineSimilarity(enrollmentEmbed, verificationEmbed);
    const isMe = similarity > THRESHOLD;
    
    // --- Step 5: Display Results ---
    printSeparator();
    console.log("🎤 HASIL VERIFIKASI SUARA");
    printSeparator();
    console.log(`Skor Kesamaan (Cosine Similarity): ${similarity.toFixed(4)}`);
    console.log(`Threshold: ${THRESHOLD}`);
    console.log(`Confidence Level: ${(similarity * 100).toFixed(2)}%`);
    
    let confidence;
    if (similarity > 0.75) {
      confidence = "SANGAT TINGGI ✅";
    } else if (similarity > 0.65) {
      confidence = "TINGGI ✓";
    } else if (similarity > 0.55) {
      confidence = "SEDANG ⚠️";
    } else {
      confidence = "RENDAH ❌";
    }
    
    console.log(`Tingkat Keyakinan: ${confidence}`);
    console.log(`\nPrediksi: ${isMe ? '✅ PEMBICARA YANG SAMA' : '❌ PEMBICARA BERBEDA'}`);
    printSeparator();
    
    if (isMe) {
      console.log("\n✅ KESIMPULAN: Suara ini terverifikasi sebagai 'isMe' (True)");
      console.log("   Suara Anda cocok dengan audio pendaftaran!");
    } else {
      console.log("\n❌ KESIMPULAN: Suara ini TIDAK terverifikasi sebagai 'isMe' (False)");
      console.log("   Suara Anda tidak cocok dengan audio pendaftaran.");
      console.log("\n💡 Kemungkinan penyebab:");
      console.log("   • Anda bukan orang yang sama dengan enrollment");
      console.log("   • Kondisi rekaman berbeda (noise, jarak mikrofon)");
      console.log("   • Intonasi atau gaya bicara berbeda");
    }
    
    console.log(`\n📌 Interpretasi Skor:`);
    if (similarity > 0.80) {
      console.log("   → Hampir pasti orang yang sama");
    } else if (similarity > 0.70) {
      console.log("   → Kemungkinan besar orang yang sama");
    } else if (similarity > 0.60) {
      console.log("   → Ada kemiripan, tapi tidak konklusif");
    } else if (similarity > 0.50) {
      console.log("   → Kemiripan rendah");
    } else {
      console.log("   → Kemungkinan besar orang yang berbeda");
    }
    
    printSeparator();
    console.log("✨ Verifikasi selesai!");
    printSeparator();
    
  } catch (error) {
    console.error(`\n❌ ERROR: ${error.message}`);
    console.error(error.stack);
    
    if (error.message.includes('tidak ditemukan')) {
      console.log("   Pastikan file audio ada di direktori yang sama dengan script ini.");
    } else if (error.code === 'ENOENT') {
      console.log("\n💡 Troubleshooting:");
      console.log("   1. Pastikan ffmpeg terinstall");
      console.log("   2. Install dependencies: npm install node-wav");
    } else {
      console.log("\n💡 Troubleshooting:");
      console.log("   1. Pastikan mikrofon terhubung dan berfungsi");
      console.log("   2. Install dependencies: npm install");
      console.log("   3. npm install node-wav");
      console.log("   4. Pastikan ffmpeg terinstall di sistem Anda");
    }
    
    process.exit(1);
  } finally {
    rl.close();
  }
}

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
  console.log("\n\n⚠️  Rekaman dibatalkan oleh user.");
  rl.close();
  process.exit(0);
});

// Run the program
main();