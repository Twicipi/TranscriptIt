# 🎙️ Whisper Transkriptor Studio

CLI & GUI Tool berbasis Python untuk transkripsi file audio dan video berdurasi panjang (hingga 8GB / 10 jam) menggunakan **faster-whisper** (CTranslate2 engine). Dilengkapi deteksi otomatis GPU CUDA / CPU, VAD filter, word-level timestamps, dan export subtitle setara kualitas YouTube.

---

## 📌 Fitur Utama

- ⚡ **Deteksi Hardware Otomatis**: Mendeteksi NVIDIA GPU (CUDA) secara otomatis dengan compute type `int8_float16` untuk efisiensi VRAM 2x, atau fallback ke CPU dengan `int8`.
- ⏱️ **Skala Besar**: Mampu memproses file rekaman hingga 8GB / 10 jam (MP4, MKV, MP3, M4A, WAV, FLAC, dll) dengan Smart Chunking + Overlap 2 detik.
- 🌐 **Bahasa Indonesia Default**: Optimal untuk transkripsi percakapan, kuliah, dan seminar. Mendukung juga deteksi otomatis (`--language auto`).
- 🧹 **VAD Filter (Silero VAD)**: Otomatis mengabaikan bagian hening/musik latar (`min_silence_duration_ms=500`).
- 📝 **Word-Level Timestamps**: Setiap kata memiliki timestamp presisi untuk subtitle setara YouTube.
- 🎬 **Subtitle YouTube-Style (.srt)**: Maks 42 karakter/baris, 2 baris/blok, pecah otomatis pada jeda > 0.75 detik atau tanda baca akhir kalimat.
- 📖 **TXT Paragraf Alami**: Segmen berdekatan digabung; paragraf baru hanya ketika jeda hening > 2.5 detik.
- 💾 **Dual Output Format**: `.txt` (paragraf alami) + `.srt` (subtitle YouTube-style), autosave per chunk.
- 🛡️ **Penanganan Error Terstruktur**: FFmpeg check, VRAM advisory, CUDA OOM handler.

---

## 🛠️ Prasyarat Sistem

1. **Python 3.10+** (Direkomendasikan Python 3.10 atau 3.11).
2. **FFmpeg terpasang di sistem PATH**:
   - **Windows** (via winget):
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```
   - **macOS** (Homebrew):
     ```bash
     brew install ffmpeg
     ```

---

## 📦 Panduan Instalasi

### 1. Buat Virtual Environment (Opsional tapi Direkomendasikan)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalasi PyTorch (untuk akselerasi GPU CUDA)
- **Jika Menggunakan GPU NVIDIA (CUDA)**:
  Kunjungi [pytorch.org](https://pytorch.org/) atau gunakan perintah berikut (contoh untuk CUDA 12.1):
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- **Jika Hanya Menggunakan CPU**:
  ```powershell
  pip install torch
  ```

### 3. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

> **Catatan**: `requirements.txt` v2 menggunakan `faster-whisper>=1.1.0` menggantikan `openai-whisper`. Engine baru ini 2–4x lebih cepat dan 50% lebih hemat VRAM berkat CTranslate2 + quantization int8.

### Transkripsi Google Gemini Audio

GUI juga menyediakan pilihan **Google Gemini 4 Hydrogen Audio**. Model ini mengunggah media ke Google AI dan membutuhkan API key, tetapi tidak memakai VRAM lokal.

1. Buat API key di [Google AI Studio](https://aistudio.google.com/apikey).
2. Pilih model Gemini pada dropdown GUI.
3. Isi kolom **API Key Gemini**, atau set environment variable sebelum menjalankan aplikasi:
   ```powershell
   $env:GEMINI_API_KEY = "API_KEY_ANDA"
   python app_gui.py
   ```
4. Instal dependensinya jika belum:
   ```powershell
   pip install google-genai
   ```

Catatan: aplikasi menggunakan `gemini-4-hydrogen` sebagai default. Model ini mendukung input audio; ketersediaan dan kuota tetap mengikuti akun Google AI/API, jadi pergantian model tidak menjamin pesan high demand tidak muncul. Model dapat diganti melalui environment variable `GEMINI_MODEL`.

#### Membedakan Pembicara

Untuk Gemini, salin `.example.env` menjadi `.env`, lalu isi nama pembicara:

```env
GEMINI_API_KEY=API_KEY_ANDA
GEMINI_MODEL=gemini-4-hydrogen
SPEAKER_A_NAME=Nama Pembicara A
SPEAKER_B_NAME=Nama Pembicara B
```

Gemini akan memberi label `[Nama Pembicara A]` dan `[Nama Pembicara B]` pada pergantian giliran bicara. Hasilnya tetap disimpan ke TXT dan SRT. `.env` tidak boleh diunggah ke GitHub; gunakan `.example.env` sebagai template.

---

## 🖥️ Cara Penggunaan via GUI (Desktop Modern)

Untuk pengalaman yang lebih mudah dan interaktif dengan tampilan antarmuka grafis (GUI):

1. **Jalankan via Script Batch (Cukup Klik 2x)**:
   - Double-click file `jalankan_ui.bat` atau `run_gui.bat`.
2. **Atau jalankan via Terminal / PowerShell**:
   ```powershell
   python app_gui.py
   ```

### Fitur GUI:
- 📁 **File Picker & Metadata Inspector**: Otomatis mendeteksi ukuran file, estimasi durasi, codec audio.
- ⚙️ **Pengaturan Lengkap**: Model (`tiny` s/d `large-v3`, termasuk `large-v3-turbo`), Bahasa, Perangkat, Mode Tugas, Smart Chunking.
- 🚀 **Progress Bar & Real-Time Transcript**: Memantau progress per chunk dengan jumlah kata per segmen.
- 💾 **Aksi Cepat 1-Klik**: Buka `.txt`, `.srt`, folder hasil, salin teks ke clipboard.
- 🛡️ **Thread Asinkron**: UI tetap responsif, tidak pernah freeze saat proses berjalan.

---

## 🚀 Cara Penggunaan (CLI)

### 1. Transkripsi Standar (Model `base`, Bahasa Indonesia)
```powershell
python transcribe.py rekaman_kuliah.mp4
```
Hasil tersimpan otomatis di folder yang sama: `.txt` dan `.srt`.

---

### 2. Memilih Model Whisper

| Model | Kebutuhan VRAM | Kecepatan | Rekomendasi |
| :--- | :---: | :---: | :--- |
| `tiny` | ~1 GB | ~32x | Cepat, audio jernih |
| `base` *(default)* | ~1 GB | ~16x | Seimbang kecepatan & akurasi |
| `small` | ~2 GB | ~6x | Direkomendasikan GPU 4GB |
| `medium` | ~5 GB | ~2x | Akurasi tinggi (GPU >= 6GB) |
| **`large-v3-turbo`** ⭐ | **~4 GB** | **~8x** | **Rekomendasi utama GPU >= 4GB** |
| `large-v3` / `large` | ~10 GB | ~1x | Akurasi maksimal profesional |

```powershell
# Rekomendasi: large-v3-turbo (akurasi large-v3, VRAM seperti medium)
python transcribe.py seminar.mkv --model large-v3-turbo

# GPU VRAM terbatas (4GB)
python transcribe.py seminar.mkv --model small

# CPU saja
python transcribe.py rekaman.mp4 --model base --device cpu
```

---

### 3. Verbose Streaming Real-Time
```powershell
python transcribe.py podcast.mp3 --model small --verbose
```

### 4. Direktori Output Kustom
```powershell
python transcribe.py video.mp4 --output-dir ./hasil_transkripsi
```

### 5. Deteksi Bahasa Otomatis
```powershell
python transcribe.py webinar.mp4 --language auto
```

### 6. Initial Prompt (Istilah Khusus / Nama Tokoh)
```powershell
python transcribe.py kuliah_ai.mp4 --initial-prompt "Topik: Jaringan Saraf Tiruan, Backpropagation, Telkom University, WGTIK."
```

---

## 📂 Struktur Proyek

```text
transkriptor/
|
+-- app_gui.py           # Antarmuka grafis desktop (Tkinter) — GUI Studio
+-- transcribe.py        # CLI utama & argument parser
+-- transcriber.py       # Engine faster-whisper, Smart Chunking + Overlap 2s, VAD
+-- utils.py             # FFmpeg checker, probe media, YouTube-SRT exporter, TXT exporter
+-- requirements.txt     # Dependensi Python
+-- jalankan_ui.bat      # Launcher GUI Windows
+-- README.md            # Dokumentasi
```

---

## ❓ Penanganan Masalah (Troubleshooting)

1. **`CUDA OutOfMemoryError`**:
   - **Solusi**: Gunakan `--model large-v3-turbo` (4GB VRAM, akurasi setara large-v3), atau `--model small`, atau `--device cpu`.

2. **`UnicodeEncodeError: 'charmap'` di Windows**:
   - Sudah ditangani otomatis dengan rekonfigurasi `sys.stdout` ke `UTF-8`.

3. **`'ffmpeg' tidak ditemukan di system PATH`**:
   - Instal FFmpeg dan tambahkan ke PATH, atau letakkan `ffmpeg.exe` langsung di folder project.

4. **Subtitle (.srt) tampil tidak tepat waktu**:
   - Gunakan model `large-v3-turbo` atau `large-v3` untuk word-level timestamps yang lebih presisi.
   - Tambahkan `--initial-prompt` dengan konteks topik untuk meningkatkan akurasi transkripsi.
