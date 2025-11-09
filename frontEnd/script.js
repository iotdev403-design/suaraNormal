document.addEventListener("DOMContentLoaded", () => {
  // --- Elemen untuk Perekaman Utama ---
  const recordButton = document.getElementById("recordButton");
  const recordingStatus = document.getElementById("recordingStatus");
  const initialTranscriptionElem = document.getElementById(
    "initialTranscription"
  );
  const finalSoundElem = document.getElementById("finalSound");
  const finalAudioElem = document.getElementById("finalAudio");
  const resultsDiv = document.getElementById("results");
  const initialAudioElem = document.getElementById("initialAudio");
  const modelSelector = document.getElementById("modelSelector");
  const promptSelector = document.getElementById("promptSelector");

  // --- BARU: Elemen untuk Perekaman Pendaftaran ---
  const enrollButton = document.getElementById("enrollButton");
  const enrollmentStatus = document.getElementById("enrollmentStatus");

  // --- Konfigurasi API ---
  let API_BASE_URL = "http://localhost:8001"; // Default ke localhost

  // --- State Perekaman Utama ---
  let isRecording = false;
  let mediaRecorder;
  let audioChunks = [];

  // --- BARU: State untuk Perekaman Pendaftaran ---
  let isEnrolling = false;
  let enrollmentRecorder;
  let enrollmentAudioChunks = [];

  // --- Stream Media Global (untuk dipakai bersama) ---
  let mediaStream = null;

  // Fungsi untuk memastikan URL API yang benar
  async function checkApiUrl() {
    // Coba gunakan URL publik jika memungkinkan
    const publicApiUrl = "http://0.0.0.0:8001";
    try {
      // Menggunakan AbortController untuk timeout cepat
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 detik timeout
      await fetch(publicApiUrl, { method: "HEAD", signal: controller.signal });
      clearTimeout(timeoutId);
      API_BASE_URL = publicApiUrl;
      console.log(`Using remote API: ${API_BASE_URL}`);
    } catch (error) {
      console.warn(`Remote API unreachable. Falling back to: ${API_BASE_URL}`);
    }
  }
  checkApiUrl(); // Jalankan pengecekan saat halaman dimuat

  // Fungsi untuk inisialisasi akses mikrofon (dipakai bersama)
  async function initializeAudio() {
    if (mediaStream && mediaStream.active) {
      return true;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      return true;
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert(
        "Could not access microphone. Please allow microphone access in your browser settings."
      );
      return false;
    }
  }

  // --- LOGIKA UNTUK PENDAFTARAN SUARA (BARU) ---
  enrollButton.addEventListener("click", async () => {
    if (isEnrolling) {
      enrollmentRecorder.stop();
      // UI akan diupdate di event 'onstop'
    } else {
      const audioInitialized = await initializeAudio();
      if (!audioInitialized) return;

      enrollmentAudioChunks = []; // Kosongkan chunk untuk rekaman baru
      enrollmentRecorder = new MediaRecorder(mediaStream, {
        mimeType: "audio/webm",
      });

      enrollmentRecorder.ondataavailable = (event) => {
        enrollmentAudioChunks.push(event.data);
      };

      enrollmentRecorder.onstop = sendEnrollmentAudio; // Kirim audio saat berhenti

      enrollmentRecorder.start();
      isEnrolling = true;
      enrollButton.classList.add("recording");
      enrollButton.textContent = "Stop Recording";
      enrollmentStatus.textContent = "Recording your voice...";
    }
  });

  async function sendEnrollmentAudio() {
    // Update UI segera setelah berhenti
    isEnrolling = false;
    enrollButton.classList.remove("recording");
    enrollButton.textContent = "Record Voice";
    enrollmentStatus.textContent = "Processing...";

    const audioBlob = new Blob(enrollmentAudioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("enrollment_file", audioBlob, "enrollment.webm");

    try {
      const response = await fetch(`${API_BASE_URL}/enroll_voice`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || "Failed to enroll voice.");
      }

      enrollmentStatus.textContent = "Voice registered successfully!";
      console.log(result.message);
    } catch (error) {
      console.error("Error uploading enrollment audio:", error);
      enrollmentStatus.textContent = `Error: ${error.message}`;
    }
  }

  // --- LOGIKA UNTUK PEREKAMAN UTAMA (SUDAH ADA, SEDIKIT DIMODIFIKASI) ---
  recordButton.addEventListener("click", async () => {
    if (isRecording) {
      mediaRecorder.stop();
    } else {
      const audioInitialized = await initializeAudio();
      if (!audioInitialized) return;

      audioChunks = []; // Kosongkan chunk untuk rekaman baru
      mediaRecorder = new MediaRecorder(mediaStream, {
        mimeType: "audio/webm",
      });

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = sendAudioToServer;
      mediaRecorder.start();

      isRecording = true;
      recordButton.classList.add("recording");
      recordingStatus.textContent = "Recording... Press to stop.";
    }
  });

  async function sendAudioToServer() {
    // Update UI
    isRecording = false;
    recordButton.classList.remove("recording");
    recordingStatus.textContent = "Processing...";

    const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
    const selectedModel = modelSelector.value;
    const selectedPrompt = promptSelector.value;

    const formData = new FormData();
    formData.append("audio_file", audioBlob, "recording.webm");
    formData.append("model_selection", selectedModel);
    formData.append("prompt_selection", selectedPrompt);

    // Reset tampilan hasil
    resultsDiv.classList.remove("hidden");
    initialTranscriptionElem.textContent = "Transcribing...";
    finalSoundElem.textContent = "Refining...";
    finalAudioElem.src = "";
    initialAudioElem.src = "";

    try {
      const processResponse = await fetch(`${API_BASE_URL}/process_audio`, {
        method: "POST",
        body: formData,
      });

      const data = await processResponse.json();

      if (!processResponse.ok) {
        // Tangani error dari server secara spesifik
        throw new Error(
          data.message || `Server error: ${processResponse.status}`
        );
      }

      initialTranscriptionElem.textContent = data.initial_transcription;
      finalSoundElem.textContent = data.natural_text;

      // Ambil file audio secara paralel
      const [summaryAudioResponse, transcriptionAudioResponse] =
        await Promise.all([
          fetch(`${API_BASE_URL}/get_response_audio`),
          fetch(`${API_BASE_URL}/get_transcription_audio`),
        ]);

      if (summaryAudioResponse.ok) {
        const audioBlob = await summaryAudioResponse.blob();
        finalAudioElem.src = URL.createObjectURL(audioBlob);
      } else {
        console.error("Could not fetch the summary audio file.");
        finalSoundElem.textContent += " (Audio failed)";
      }

      if (transcriptionAudioResponse.ok) {
        const audioBlob = await transcriptionAudioResponse.blob();
        initialAudioElem.src = URL.createObjectURL(audioBlob);
      } else {
        console.error("Could not fetch the transcription audio file.");
        initialTranscriptionElem.textContent += " (Audio failed)";
      }

      recordingStatus.textContent = "Done! Record another?";
    } catch (error) {
      console.error("Error processing audio:", error);
      initialTranscriptionElem.textContent = "An error occurred.";
      finalSoundElem.textContent = "Please check the console for details.";
      recordingStatus.textContent = `ERROR: ${error.message}`;
    }
  }
});
