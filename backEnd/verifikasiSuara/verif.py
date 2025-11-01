import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import librosa
import soundfile as sf
import pyaudio
import wave
import threading
import sys

# --- 1. Persiapan Awal ---
# Install library yang diperlukan:
# pip install resemblyzer
# pip install librosa soundfile pyaudio

# Teks narasi untuk referensi
narasi = """
Di tanah kuno Eldoria, di mana langit berkilau dan hutan berbisik rahasia kepada angin, 
hiduplah seekor naga bernama Zephyros. [sarcastically] Bukan tipe yang "membakar semuanya"… 
[giggles] tapi dia lembut, bijaksana, dengan mata seperti bintang tua. [whispers] 
Bahkan burung-burung pun terdiam saat dia lewat.
"""

# Nama file audio untuk enrollment (pendaftaran)
nama_file_audio = "audio_saya.mp3"

print("🎙️  SISTEM VERIFIKASI SUARA DENGAN RESEMBLYZER")
print("=" * 60)

# --- 2. Buat File Enrollment dari Audio Asli ---
try:
    print(f"\n📂 Memuat audio enrollment dari '{nama_file_audio}'...")
    
    # Load audio menggunakan librosa (support MP3, WAV, dll)
    audio_data, sample_rate = librosa.load(nama_file_audio, sr=None)
    
    durasi_total = len(audio_data) / sample_rate
    print(f"✅ Audio berhasil dimuat!")
    print(f"   Durasi: {durasi_total:.2f} detik")
    print(f"   Sample Rate: {sample_rate} Hz")
    
    # Validasi durasi minimum
    if durasi_total < 10:
        print(f"⚠️  PERINGATAN: Audio terlalu pendek ({durasi_total:.2f}s).")
        print(f"   Minimal 10 detik direkomendasikan untuk hasil optimal.")
    
    # Gunakan 12 detik pertama untuk enrollment (atau seluruh audio jika < 12 detik)
    titik_pisah_detik = min(12, durasi_total)
    titik_pisah_sample = int(titik_pisah_detik * sample_rate)
    
    # Buat file untuk pendaftaran (enrollment)
    audio_enrollment = audio_data[:titik_pisah_sample]
    sf.write("enrollment.wav", audio_enrollment, sample_rate)
    
    durasi_enrollment = len(audio_enrollment) / sample_rate
    
    print(f"\n✅ File enrollment dibuat:")
    print(f"   📝 enrollment.wav: {durasi_enrollment:.2f} detik")

except FileNotFoundError:
    print(f"\n❌ ERROR: File '{nama_file_audio}' tidak ditemukan!")
    print("   Pastikan file audio ada di direktori yang sama dengan script ini.")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR saat memproses audio: {e}")
    exit(1)

# --- 3. Fungsi Perekaman Audio dari Mikrofon ---
def rekam_audio(filename, durasi=10, sample_rate=16000):
    """
    Merekam audio dari mikrofon
    
    Args:
        filename: Nama file output
        durasi: Durasi rekaman dalam detik
        sample_rate: Sample rate audio
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    
    try:
        # Buka stream audio
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"\n🔴 REKAMAN DIMULAI! (Durasi: {durasi} detik)")
        print(f"   Silakan berbicara sekarang...")
        print(f"   💡 Tip: Bacakan narasi yang sama untuk hasil terbaik")
        
        frames = []
        
        # Hitung jumlah chunk yang dibutuhkan
        total_chunks = int(sample_rate / CHUNK * durasi)
        
        # Progress bar sederhana
        for i in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Progress indicator
            progress = (i + 1) / total_chunks * 100
            bars = int(progress / 5)
            sys.stdout.write(f"\r   Progress: [{'█' * bars}{'░' * (20 - bars)}] {progress:.0f}%")
            sys.stdout.flush()
        
        print(f"\n✅ Rekaman selesai!")
        
        # Stop stream
        stream.stop_stream()
        stream.close()
        
        # Simpan ke file WAV
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"   💾 Audio disimpan ke '{filename}'")
        
    except Exception as e:
        print(f"\n❌ ERROR saat merekam: {e}")
        raise
    finally:
        p.terminate()

# --- 4. Rekam Audio Verification dari Mikrofon ---
try:
    print("\n" + "=" * 60)
    print("🎤 TAHAP VERIFIKASI - REKAM SUARA ANDA")
    print("=" * 60)
    
    # Tanya durasi rekaman
    print("\nBerapa detik Anda ingin merekam? (default: 10 detik)")
    print("Tekan Enter untuk menggunakan default")
    
    durasi_input = input("Durasi (detik): ").strip()
    durasi_rekam = int(durasi_input) if durasi_input else 10
    
    if durasi_rekam < 3:
        print("⚠️  Durasi terlalu pendek. Minimal 3 detik. Menggunakan 5 detik.")
        durasi_rekam = 5
    elif durasi_rekam > 30:
        print("⚠️  Durasi terlalu panjang. Maksimal 30 detik.")
        durasi_rekam = 30
    
    print(f"\n📌 Bersiap untuk merekam selama {durasi_rekam} detik...")
    input("   Tekan ENTER untuk mulai merekam...")
    
    # Rekam audio
    rekam_audio("verification.wav", durasi=durasi_rekam, sample_rate=16000)

except KeyboardInterrupt:
    print("\n\n⚠️  Rekaman dibatalkan oleh user.")
    exit(0)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Pastikan mikrofon terhubung dan berfungsi")
    print("   2. Coba: pip install --upgrade pyaudio")
    print("   3. Di Linux: sudo apt-get install portaudio19-dev python3-pyaudio")
    print("   4. Di Mac: brew install portaudio")
    exit(1)

# --- 5. Inisialisasi Voice Encoder ---
try:
    print(f"\n🤖 Memuat model Resemblyzer...")
    
    encoder = VoiceEncoder()
    
    print("✅ Model berhasil dimuat!")

except Exception as e:
    print(f"\n❌ ERROR saat memuat model: {e}")
    exit(1)

# --- 6. Preprocess Audio ---
try:
    print(f"\n🔧 Memproses audio untuk verifikasi...")
    
    # Preprocess kedua file audio
    enrollment_wav = preprocess_wav(Path("enrollment.wav"))
    verification_wav = preprocess_wav(Path("verification.wav"))
    
    print("✅ Audio berhasil diproses!")

except Exception as e:
    print(f"\n❌ ERROR saat preprocessing audio: {e}")
    exit(1)

# --- 7. Ekstrak Voice Embeddings ---
try:
    print(f"\n🧬 Mengekstrak fitur suara (voice embeddings)...")
    
    # Ekstrak embedding dari kedua audio
    enrollment_embed = encoder.embed_utterance(enrollment_wav)
    verification_embed = encoder.embed_utterance(verification_wav)
    
    print(f"✅ Embedding berhasil diekstrak!")
    print(f"   Dimensi embedding: {enrollment_embed.shape[0]}D")

except Exception as e:
    print(f"\n❌ ERROR saat ekstraksi embedding: {e}")
    exit(1)

# --- 8. Hitung Similarity ---
try:
    print(f"\n📊 Menghitung kesamaan suara...")
    
    # Cosine similarity antara dua embedding
    similarity = np.dot(enrollment_embed, verification_embed)
    
    # Threshold untuk menentukan apakah sama atau beda
    threshold = 0.70
    
    isMe = similarity > threshold
    
    # --- 9. Tampilkan Hasil ---
    print("\n" + "=" * 60)
    print("🎤 HASIL VERIFIKASI SUARA")
    print("=" * 60)
    print(f"Skor Kesamaan (Cosine Similarity): {similarity:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Confidence Level: {similarity * 100:.2f}%")
    
    if similarity > 0.75:
        confidence = "SANGAT TINGGI ✅"
    elif similarity > 0.65:
        confidence = "TINGGI ✓"
    elif similarity > 0.55:
        confidence = "SEDANG ⚠️"
    else:
        confidence = "RENDAH ❌"
    
    print(f"Tingkat Keyakinan: {confidence}")
    print(f"\nPrediksi: {'✅ PEMBICARA YANG SAMA' if isMe else '❌ PEMBICARA BERBEDA'}")
    print("=" * 60)
    
    if isMe:
        print("\n✅ KESIMPULAN: Suara ini terverifikasi sebagai 'isMe' (True)")
        print("   Suara Anda cocok dengan audio pendaftaran!")
    else:
        print("\n❌ KESIMPULAN: Suara ini TIDAK terverifikasi sebagai 'isMe' (False)")
        print("   Suara Anda tidak cocok dengan audio pendaftaran.")
        print("\n💡 Kemungkinan penyebab:")
        print("   • Anda bukan orang yang sama dengan enrollment")
        print("   • Kondisi rekaman berbeda (noise, jarak mikrofon)")
        print("   • Intonasi atau gaya bicara berbeda")
    
    # Interpretasi tambahan
    print(f"\n📌 Interpretasi Skor:")
    if similarity > 0.80:
        print("   → Hampir pasti orang yang sama")
    elif similarity > 0.70:
        print("   → Kemungkinan besar orang yang sama")
    elif similarity > 0.60:
        print("   → Ada kemiripan, tapi tidak konklusif")
    elif similarity > 0.50:
        print("   → Kemiripan rendah")
    else:
        print("   → Kemungkinan besar orang yang berbeda")

except Exception as e:
    print(f"\n❌ ERROR saat verifikasi: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✨ Verifikasi selesai!")
print("=" * 60)