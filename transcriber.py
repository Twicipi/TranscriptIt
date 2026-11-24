"""
Module: transcriber.py
Deskripsi: Engine transkripsi berbasis faster-whisper (CTranslate2) dengan deteksi hardware
           pintar (CUDA/CPU, int8_float16/int8), mitigasi memory error, Smart Chunking
           dengan overlap 2 detik, VAD filter, dan word-level timestamps untuk output
           subtitle setara kualitas YouTube.
"""

import gc
import math
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils import Colors, format_duration, format_file_size, print_info, print_warning


# ---------------------------------------------------------------------------
# Daftar model yang didukung faster-whisper
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large",
    "large-v3-turbo",
    "turbo",
]

# Estimasi kebutuhan VRAM (GB) — digunakan untuk advisory warning
MODEL_VRAM_ESTIMATES: Dict[str, float] = {
    "tiny":           1.0,
    "base":           1.0,
    "small":          2.0,
    "medium":         5.0,
    "large":          10.0,
    "large-v1":       10.0,
    "large-v2":       10.0,
    "large-v3":       10.0,
    "large-v3-turbo": 4.0,
    "turbo":          4.0,
}

# Overlap antar-chunk (detik) untuk mencegah kata terpotong di batas chunk
CHUNK_OVERLAP_SEC: float = 2.0

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# ---------------------------------------------------------------------------
# 1. DETEKSI HARDWARE
# ---------------------------------------------------------------------------

def detect_optimal_device(
    requested_device: str = "auto",
    force_fp32: bool = False
) -> Tuple[str, bool, str]:
    """
    Mendeteksi hardware akselerasi (GPU CUDA vs CPU) dan menentukan flag FP16.
    Returns:
        (device_str, use_fp16, description_str)
    """
    try:
        import torch
    except ImportError:
        print_warning(
            "Library 'torch' belum terpasang. Gunakan 'pip install torch' untuk melanjutkan."
        )
        return "cpu", False, "CPU (torch not loaded)"

    requested = requested_device.lower().strip()

    if requested == "cpu":
        return "cpu", False, "CPU (Dipaksa oleh pengguna)"

    cuda_available = torch.cuda.is_available()

    if requested == "cuda" and not cuda_available:
        print_warning(
            "Perangkat 'cuda' diminta namun GPU NVIDIA / CUDA tidak terdeteksi. "
            "Beralih secara otomatis ke mode CPU."
        )
        return "cpu", False, "CPU (Fallback karena CUDA tidak tersedia)"

    if cuda_available and requested in ["auto", "cuda"]:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        vram_str = format_file_size(total_vram)
        fp16 = not force_fp32
        mode_fp = "FP16" if fp16 else "FP32 (Forced)"
        desc = f"NVIDIA GPU: {gpu_name} ({vram_str} VRAM) [{mode_fp}]"
        return device, fp16, desc
    else:
        return "cpu", False, "CPU Standard (CUDA tidak terdeteksi)"


# ---------------------------------------------------------------------------
# 2. ADVISORY VRAM
# ---------------------------------------------------------------------------

def check_vram_advisory(model_name: str, device: str) -> None:
    """Peringatan dini jika model melebihi kapasitas VRAM GPU."""
    if device != "cuda":
        return
    try:
        import torch
        base_name = model_name.split(".")[0]
        required_vram = MODEL_VRAM_ESTIMATES.get(base_name, 2.0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if required_vram > total_vram_gb:
            print_warning(
                f"Model '{model_name}' diperkirakan membutuhkan ~{required_vram:.1f} GB VRAM, "
                f"sedangkan kapasitas GPU Anda adalah {total_vram_gb:.1f} GB.\n"
                f"   💡 Jika terjadi 'CUDA OutOfMemory', gunakan model yang lebih ringan "
                f"seperti '--model small', '--model large-v3-turbo', atau tambahkan '--device cpu'."
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 3. LOAD MODEL (faster-whisper)
# ---------------------------------------------------------------------------

def select_optimal_compute_type(device: str = "cpu", force_fp32: bool = False) -> str:
    """
    Memilih compute_type terbaik yang benar-benar didukung oleh arsitektur GPU/CPU saat ini.
    - RTX 20/30/40, GTX 16xx (Turing/Ampere/Ada) : int8_float16
    - GTX 1050/1060/1070/1080 (Pascal)          : int8_float32 / int8
    - CPU                                       : int8
    """
    if force_fp32:
        return "float32"

    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types(device)
    except Exception:
        return "int8" if device == "cpu" else "float32"

    if device == "cuda":
        for ct in ["int8_float16", "float16", "int8_float32", "int8", "float32"]:
            if ct in supported:
                return ct
        return "float32"
    else:
        for ct in ["int8", "int8_float32", "float32"]:
            if ct in supported:
                return ct
        return "int8"


def load_whisper_model(
    model_name: str = "base",
    device: str = "cpu",
    force_fp32: bool = False,
    download_root: Optional[str] = None,
) -> Any:
    """
    Memuat model faster-whisper ke perangkat yang ditargetkan dengan pemilihan
    compute_type otomatis yang kompatibel dengan GPU/CPU.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "Library 'faster-whisper' belum terpasang!\n"
            "Silakan install: pip install faster-whisper>=1.1.0"
        )

    clean_model = model_name.lower().strip()
    if clean_model not in AVAILABLE_MODELS and not Path(clean_model).exists():
        print_warning(
            f"Nama model '{model_name}' tidak ada di daftar resmi {AVAILABLE_MODELS}. "
            "Mencoba memuat langsung..."
        )

    check_vram_advisory(clean_model, device)

    # Tentukan compute_type secara otomatis berdasarkan dukungan perangkat
    compute_type = select_optimal_compute_type(device=device, force_fp32=force_fp32)

    print_info(
        "Model Loading",
        f"Memuat faster-whisper '{clean_model}' ke {device.upper()} "
        f"[compute_type={compute_type}]..."
    )
    start_time = time.time()

    kwargs: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }
    if download_root:
        kwargs["download_root"] = download_root

    try:
        model = WhisperModel(clean_model, **kwargs)
    except ValueError as e:
        err_msg = str(e).lower()
        if "compute type" in err_msg and device == "cuda":
            print_warning(f"Compute type '{compute_type}' ditolak GPU. Mencoba fallback kompatibel...")
            for fallback_ct in ["int8_float32", "int8", "float32"]:
                if fallback_ct == compute_type:
                    continue
                try:
                    kwargs["compute_type"] = fallback_ct
                    model = WhisperModel(clean_model, **kwargs)
                    compute_type = fallback_ct
                    print_info("Fallback Sukses", f"Berhasil memuat model dengan compute_type={compute_type}")
                    break
                except Exception:
                    continue
            else:
                raise e
        else:
            raise e

    elapsed = time.time() - start_time
    print_info("Model Ready", f"Model siap dalam {elapsed:.2f} detik.\n")
    return model


def transcribe_media_gemini(
    file_path: Path,
    language: Optional[str] = "id",
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    total_duration: float = 0.0,
    speaker_a_name: str = "Si A",
    speaker_b_name: str = "Si B",
) -> Dict[str, Any]:
    """Transcribe a media file with Gemini's native audio understanding."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "SDK Gemini belum terpasang. Jalankan: pip install google-genai"
        ) from exc

    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "API key Gemini belum diisi. Masukkan di kolom API Key atau set GEMINI_API_KEY."
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"File media tidak ditemukan: {file_path}")

    language_name = language or "bahasa audio"
    task_instruction = (
        "Transkripsikan audio apa adanya."
        if task != "translate"
        else "Transkripsikan audio lalu terjemahkan hasilnya ke bahasa Inggris."
    )
    prompt = (
        "Anda adalah mesin transkripsi audio. "
        f"{task_instruction} Gunakan {language_name} bila bahasa ditentukan. "
        "Bedakan dua pembicara berdasarkan karakteristik suaranya. "
        f"Pembicara pertama harus diberi label [{speaker_a_name}] dan pembicara kedua [{speaker_b_name}]. "
        f"Awali setiap giliran bicara dengan label [{speaker_a_name}] atau [{speaker_b_name}] "
        "pada baris baru. Jika tidak yakin, pilih pembicara yang paling konsisten. "
        "Keluarkan hanya teks transkripsi tanpa judul, markdown, timestamp, atau komentar tambahan."
    )
    if initial_prompt:
        prompt += f" Istilah/nama yang perlu dipertahankan: {initial_prompt}."

    client = genai.Client(api_key=key)
    uploaded_file = None
    try:
        mime_type = mimetypes.guess_type(file_path.name)[0]
        uploaded_file = client.files.upload(
            file=str(file_path),
            config=types.UploadFileConfig(mime_type=mime_type) if mime_type else None,
        )
        response = None
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded_file, prompt],
                )
                break
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                is_temporary = any(
                    marker in error_text
                    for marker in ("503", "unavailable", "high demand", "resource exhausted")
                )
                if not is_temporary or attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

        if response is None and last_error is not None:
            raise last_error
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini mengembalikan transkripsi kosong.")
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    end_time = max(float(total_duration or 0.0), 0.1)
    return {
        "segments": [{"start": 0.0, "end": end_time, "text": text, "words": []}],
        "text": text,
        "language": language or "auto",
    }


# ---------------------------------------------------------------------------
# 4. EKSTRAKSI AUDIO CHUNK (dengan audio filter)
# ---------------------------------------------------------------------------

def extract_audio_chunk(
    media_path: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Mengekstrak potongan audio spesifik dari file media via FFmpeg subprocess.
    - Hemat RAM: tidak memuat keseluruhan file ke memori.
    - Filter audio: highpass 80 Hz (hilangkan dengung rendah) + dynaudnorm
      (normalisasi dinamis agar vokal lebih jelas dan konsisten).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(media_path),
        # Filter audio ringan: hapus dengung frekuensi rendah + normalisasi vokal
        "-af", "highpass=f=80,dynaudnorm=f=150:g=15",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 and len(proc.stdout) == 0:
        return np.array([], dtype=np.float32)

    # Konversi raw PCM 16-bit ke float32 [-1.0, 1.0]
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# 5. TRANSKRIPSI CHUNKED DENGAN OVERLAP & DEDUP
# ---------------------------------------------------------------------------

def transcribe_media_chunked(
    model: Any,
    file_path: Path,
    total_duration: float,
    chunk_size_sec: float = 600.0,
    language: Optional[str] = "id",
    task: str = "transcribe",
    verbose: bool = True,
    fp16: bool = True,                # Dipertahankan untuk kompatibilitas signature CLI/GUI
    initial_prompt: Optional[str] = None,
    on_chunk_completed: Optional[Callable[[int, int, List[Dict[str, Any]], str], None]] = None,
    on_segment: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk_start: Optional[Callable[[int, int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Transkripsi Smart Chunking dengan:
      - Overlap 2 detik antar chunk untuk mencegah kata terpotong.
      - Deduplication berbasis 'last_committed_end' agar segmen tidak dobel.
      - VAD filter aktif (min_silence_duration_ms=500, speech_pad_ms=250).
      - Word-level timestamps (word, start, end, probability) di tiap segmen.
      - initial_prompt murni dari pengguna (tidak di-chain ke chunk berikutnya).
      - condition_on_previous_text=False untuk mencegah hallucination drift.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File media tidak ditemukan: {file_path}")

    # Normalisasi bahasa
    lang_param: Optional[str] = (
        None if (language is None or language.lower() in ["auto", "none", ""])
        else language.lower()
    )

    if total_duration is None or total_duration <= 0:
        total_duration = 36000.0  # Fallback 10 jam

    total_chunks = max(1, math.ceil(total_duration / chunk_size_sec))
    all_segments: List[Dict[str, Any]] = []
    full_text_list: List[str] = []
    detected_language: str = lang_param or "id"

    # Batas timestamp global segmen terakhir yang sudah dikonfirmasi (untuk dedup overlap)
    last_committed_end: float = 0.0

    print(
        f"{Colors.CYAN}🧩 Memulai transkripsi dengan Smart Chunking + Overlap "
        f"({total_chunks} bagian @ {int(chunk_size_sec // 60)} menit, "
        f"overlap {CHUNK_OVERLAP_SEC:.0f}s, VAD aktif)...{Colors.RESET}"
    )
    sys.stdout.flush()

    # VAD parameters (Silero VAD via faster-whisper)
    vad_params: Dict[str, Any] = {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 250,
    }

    for chunk_idx in range(total_chunks):
        if stop_check and stop_check():
            print(f"{Colors.YELLOW}⚠️ Transkripsi dihentikan atas permintaan pengguna.{Colors.RESET}")
            break

        # Hitung rentang chunk dengan overlap
        chunk_start = chunk_idx * chunk_size_sec
        chunk_start_with_overlap = max(0.0, chunk_start - CHUNK_OVERLAP_SEC) if chunk_idx > 0 else 0.0
        chunk_end = min(chunk_start + chunk_size_sec, total_duration)
        dur_sec = chunk_end - chunk_start_with_overlap

        if dur_sec <= 0.1:
            break

        # Rentang yang ditampilkan ke pengguna (tanpa overlap)
        chunk_time_str = f"{format_duration(chunk_start)} - {format_duration(chunk_end)}"
        print(
            f"\n{Colors.BLUE}{Colors.BOLD}▶ [{chunk_idx + 1}/{total_chunks}]{Colors.RESET} "
            f"Memproses rentang {Colors.BOLD}{chunk_time_str}{Colors.RESET} "
            f"(extract: {format_duration(chunk_start_with_overlap)} - {format_duration(chunk_end)})..."
        )
        sys.stdout.flush()

        if on_chunk_start:
            try:
                on_chunk_start(chunk_idx + 1, total_chunks, chunk_time_str)
            except Exception:
                pass

        # Ekstrak audio chunk (dengan overlap di sisi kiri)
        chunk_audio = extract_audio_chunk(file_path, chunk_start_with_overlap, dur_sec)
        if len(chunk_audio) == 0:
            print_warning(f"Audio chunk {chunk_idx + 1} kosong atau mencapai akhir file.")
            break

        # ---- faster-whisper transcribe ----
        try:
            segments_iter, info = model.transcribe(
                chunk_audio,
                language=lang_param,
                task=task,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=vad_params,
            )
            # Materialize generator sebelum gc
            raw_segments = list(segments_iter)
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                raise RuntimeError(
                    f"\n[CUDA OUT OF MEMORY] Memori GPU tidak cukup pada bagian {chunk_idx + 1}. "
                    f"Gunakan model lebih kecil atau '--device cpu'."
                ) from e
            raise

        # Update bahasa terdeteksi
        if info and hasattr(info, "language") and info.language:
            detected_language = info.language

        chunk_text_parts: List[str] = []

        for seg in raw_segments:
            # Global offset timestamp (relatif terhadap keseluruhan media)
            global_start = seg.start + chunk_start_with_overlap
            global_end = seg.end + chunk_start_with_overlap

            # --- Deduplication overlap ---
            # Lewati segmen yang seluruhnya berada di dalam zona overlap yang sudah
            # diproses oleh chunk sebelumnya.
            if global_end <= last_committed_end:
                continue
            # Jika awal segmen tumpang tindih tapi akhirnya baru: geser awal ke
            # last_committed_end agar tidak ada celah maupun duplikasi.
            if global_start < last_committed_end:
                global_start = last_committed_end

            # Susun data words dengan offset global
            words_data: List[Dict[str, Any]] = []
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word":        w.word,
                        "start":       round(w.start + chunk_start_with_overlap, 3),
                        "end":         round(w.end   + chunk_start_with_overlap, 3),
                        "probability": round(w.probability, 4),
                    })

            seg_text = seg.text.strip()

            # Buat dictionary segmen seragam (kompatibel dengan CLI & GUI)
            seg_dict: Dict[str, Any] = {
                "id":          len(all_segments),
                "start":       round(global_start, 3),
                "end":         round(global_end, 3),
                "text":        seg_text,
                "words":       words_data,
                "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
            }

            all_segments.append(seg_dict)
            last_committed_end = global_end

            if seg_text:
                chunk_text_parts.append(seg_text)

            # Callback real-time ke GUI / consumer
            if on_segment:
                try:
                    on_segment(seg_dict)
                except Exception:
                    pass

            if verbose:
                t_start = format_duration(seg_dict["start"])
                t_end = format_duration(seg_dict["end"])
                print(
                    f"  {Colors.DIM}[{t_start} --> {t_end}]{Colors.RESET} "
                    f"{seg_text}"
                )
                sys.stdout.flush()

        if chunk_text_parts:
            full_text_list.extend(chunk_text_parts)

        # Autosave per chunk
        current_full_text = " ".join(full_text_list)
        if on_chunk_completed:
            try:
                on_chunk_completed(chunk_idx + 1, total_chunks, all_segments, current_full_text)
            except Exception as save_err:
                print_warning(f"Peringatan saat autosave progress: {save_err}")

        # Bersihkan memori
        del chunk_audio
        del raw_segments
        gc.collect()

    return {
        "text":     " ".join(full_text_list),
        "segments": all_segments,
        "language": detected_language,
    }
