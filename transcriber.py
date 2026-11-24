"""
Module: transcriber.py
Deskripsi: Layer Service / Business Orchestrator (MVC).
           Mengorkestrasi engine faster-whisper (CTranslate2), Gemini native audio,
           dan pipeline hybrid YouTube (CC YouTube -> fallback yt-dlp + Whisper).

Catatan Arsitektur:
  - Bebas dari print(), input(), sys.exit(), maupun referensi widget GUI.
  - Seluruh pelaporan status disuntikkan melalui progress_callback (DIP).
"""

import gc
import math
import mimetypes
import os
import shutil
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils import (
    download_youtube_audio,
    extract_video_id,
    fetch_youtube_transcript,
    format_duration,
    probe_media_file,
)


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

MODEL_VRAM_ESTIMATES: Dict[str, float] = {
    "tiny": 1.0,
    "base": 1.0,
    "small": 2.0,
    "medium": 5.0,
    "large": 10.0,
    "large-v1": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.0,
    "large-v3-turbo": 4.0,
    "turbo": 4.0,
}

CHUNK_OVERLAP_SEC: float = 2.0
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-4-hydrogen")
CUDA_DLL_HANDLES: List[Any] = []


def _notify(callback: Optional[Callable[[str], None]], message: str) -> None:
    """Helper internal untuk meneruskan pesan status ke callback jika tersedia."""
    if callback:
        try:
            callback(message)
        except Exception:
            pass


def configure_cuda_runtime() -> None:
    """Make pip-installed NVIDIA runtime DLLs visible to Windows loaders."""
    if os.name != "nt":
        return

    site_packages = [Path(path) for path in site.getsitepackages()]
    site_packages.append(Path(sys.prefix) / "Lib" / "site-packages")
    dll_dirs = {
        package_dir / "nvidia" / "cublas" / "bin"
        for package_dir in site_packages
    }
    dll_dirs.update(
        package_dir / "nvidia" / "cudnn" / "bin"
        for package_dir in site_packages
    )
    for dll_dir in dll_dirs:
        if dll_dir.is_dir():
            os.environ["PATH"] = f"{dll_dir};{os.environ.get('PATH', '')}"
            try:
                CUDA_DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            except (AttributeError, OSError):
                pass


# ---------------------------------------------------------------------------
# 1. DETEKSI HARDWARE & ADVISORY VRAM
# ---------------------------------------------------------------------------

def detect_optimal_device(
    requested_device: str = "auto",
    force_fp32: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, bool, str]:
    """
    Mendeteksi hardware akselerasi (GPU CUDA vs CPU) dan menentukan flag FP16.
    """
    requested = requested_device.lower().strip()

    if requested == "cpu":
        return "cpu", False, "CPU (Dipaksa oleh pengguna)"

    cuda_available = False
    gpu_name = "NVIDIA GPU"
    vram_text = "VRAM tidak diketahui"
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            gpu_name, vram_mib = [part.strip() for part in probe.stdout.splitlines()[0].split(",", 1)]
            vram_text = f"{vram_mib} MiB VRAM"
            cuda_available = True
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    if requested == "cuda" and not cuda_available:
        _notify(
            progress_callback,
            "Peringatan: Perangkat 'cuda' diminta namun GPU NVIDIA tidak terdeteksi. Beralih ke CPU.",
        )
        return "cpu", False, "CPU (Fallback karena CUDA tidak tersedia)"

    if cuda_available and requested in ["auto", "cuda"]:
        device = "cuda"
        fp16 = not force_fp32
        mode_fp = "FP16" if fp16 else "FP32 (Forced)"
        desc = f"NVIDIA GPU: {gpu_name} ({vram_text}) [{mode_fp}]"
        return device, fp16, desc
    return "cpu", False, "CPU Standard (CUDA tidak terdeteksi)"


def check_vram_advisory(
    model_name: str,
    device: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Peringatan dini jika model berpotensi melebihi kapasitas VRAM GPU."""
    if device != "cuda":
        return
    try:
        import torch
        base_name = model_name.split(".")[0]
        required_vram = MODEL_VRAM_ESTIMATES.get(base_name, 2.0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if required_vram > total_vram_gb:
            _notify(
                progress_callback,
                f"Peringatan VRAM: Model '{model_name}' butuh ~{required_vram:.1f} GB VRAM, "
                f"kapasitas GPU {total_vram_gb:.1f} GB. Gunakan 'small'/'large-v3-turbo' atau '--device cpu' jika OOM.",
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 2. LOAD MODEL (faster-whisper)
# ---------------------------------------------------------------------------

def select_optimal_compute_type(device: str = "cpu", force_fp32: bool = False) -> str:
    """Memilih compute_type terbaik yang didukung arsitektur GPU/CPU."""
    if force_fp32:
        return "float32"

    try:
        configure_cuda_runtime()
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types(device)
    except Exception:
        return "int8" if device == "cpu" else "float32"

    if device == "cuda":
        for ct in ["int8_float16", "float16", "int8_float32", "int8", "float32"]:
            if ct in supported:
                return ct
        return "float32"

    for ct in ["int8", "int8_float32", "float32"]:
        if ct in supported:
            return ct
    return "int8"


def load_whisper_model(
    model_name: str = "base",
    device: str = "cpu",
    force_fp32: bool = False,
    download_root: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Any:
    """Memuat model faster-whisper ke perangkat target."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "Library 'faster-whisper' belum terpasang! Jalankan: pip install faster-whisper>=1.1.0"
        ) from exc

    clean_model = model_name.lower().strip()
    if clean_model not in AVAILABLE_MODELS and not Path(clean_model).exists():
        _notify(
            progress_callback,
            f"Peringatan: Nama model '{model_name}' tidak ada di daftar standar, mencoba memuat langsung...",
        )

    check_vram_advisory(clean_model, device, progress_callback=progress_callback)
    compute_type = select_optimal_compute_type(device=device, force_fp32=force_fp32)

    _notify(
        progress_callback,
        f"Memuat faster-whisper '{clean_model}' ke {device.upper()} [compute_type={compute_type}]...",
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
            _notify(progress_callback, f"Compute type '{compute_type}' ditolak GPU, mencoba fallback...")
            for fallback_ct in ["int8_float32", "int8", "float32"]:
                if fallback_ct == compute_type:
                    continue
                try:
                    kwargs["compute_type"] = fallback_ct
                    model = WhisperModel(clean_model, **kwargs)
                    compute_type = fallback_ct
                    _notify(progress_callback, f"Fallback sukses dengan compute_type={compute_type}.")
                    break
                except Exception:
                    continue
            else:
                raise e
        else:
            raise e

    elapsed = time.time() - start_time
    _notify(progress_callback, f"Model siap dalam {elapsed:.2f} detik.")
    return model


# ---------------------------------------------------------------------------
# 3. GEMINI NATIVE AUDIO TRANSCRIPTION
# ---------------------------------------------------------------------------

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
# 4. EKSTRAKSI AUDIO CHUNK
# ---------------------------------------------------------------------------

def extract_audio_chunk(
    media_path: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Mengekstrak potongan audio spesifik dari file media via FFmpeg subprocess
    dengan highpass 80Hz + dynaudnorm menuju PCM 16-bit 16kHz mono.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(media_path),
        "-af", "highpass=f=80,dynaudnorm=f=150:g=15",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 and len(proc.stdout) == 0:
        return np.array([], dtype=np.float32)

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
    fp16: bool = True,
    initial_prompt: Optional[str] = None,
    on_chunk_completed: Optional[Callable[[int, int, List[Dict[str, Any]], str], None]] = None,
    on_segment: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk_start: Optional[Callable[[int, int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Orkestrasi transkripsi Smart Chunking + Overlap 2 detik, VAD filter,
    dan word-level timestamps.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File media tidak ditemukan: {file_path}")

    lang_param: Optional[str] = (
        None if (language is None or language.lower() in ["auto", "none", ""])
        else language.lower()
    )

    if total_duration is None or total_duration <= 0:
        total_duration = 36000.0

    total_chunks = max(1, math.ceil(total_duration / chunk_size_sec))
    all_segments: List[Dict[str, Any]] = []
    full_text_list: List[str] = []
    detected_language: str = lang_param or "id"
    last_committed_end: float = 0.0

    _notify(
        progress_callback,
        f"Memulai transkripsi Smart Chunking ({total_chunks} bagian @ {int(chunk_size_sec // 60)} menit, overlap {CHUNK_OVERLAP_SEC:.0f}s, VAD aktif)...",
    )

    vad_params: Dict[str, Any] = {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 250,
    }

    for chunk_idx in range(total_chunks):
        if stop_check and stop_check():
            _notify(progress_callback, "Transkripsi dihentikan atas permintaan pengguna.")
            break

        chunk_start = chunk_idx * chunk_size_sec
        chunk_start_with_overlap = max(0.0, chunk_start - CHUNK_OVERLAP_SEC) if chunk_idx > 0 else 0.0
        chunk_end = min(chunk_start + chunk_size_sec, total_duration)
        dur_sec = chunk_end - chunk_start_with_overlap

        if dur_sec <= 0.1:
            break

        chunk_time_str = f"{format_duration(chunk_start)} - {format_duration(chunk_end)}"
        _notify(
            progress_callback,
            f"[{chunk_idx + 1}/{total_chunks}] Memproses rentang {chunk_time_str}...",
        )

        if on_chunk_start:
            try:
                on_chunk_start(chunk_idx + 1, total_chunks, chunk_time_str)
            except Exception:
                pass

        chunk_audio = extract_audio_chunk(file_path, chunk_start_with_overlap, dur_sec)
        if len(chunk_audio) == 0:
            _notify(progress_callback, f"Audio chunk {chunk_idx + 1} kosong atau mencapai akhir file.")
            break

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
                    f"[CUDA OUT OF MEMORY] Memori GPU tidak cukup pada bagian {chunk_idx + 1}. "
                    f"Gunakan model lebih kecil atau '--device cpu'."
                ) from e
            raise

        if info and hasattr(info, "language") and info.language:
            detected_language = info.language

        chunk_text_parts: List[str] = []

        for seg in raw_segments:
            global_start = seg.start + chunk_start_with_overlap
            global_end = seg.end + chunk_start_with_overlap

            if global_end <= last_committed_end:
                continue
            if global_start < last_committed_end:
                global_start = last_committed_end

            words_data: List[Dict[str, Any]] = []
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word,
                        "start": round(w.start + chunk_start_with_overlap, 3),
                        "end": round(w.end + chunk_start_with_overlap, 3),
                        "probability": round(w.probability, 4),
                    })

            seg_text = seg.text.strip()
            seg_dict: Dict[str, Any] = {
                "id": len(all_segments),
                "start": round(global_start, 3),
                "end": round(global_end, 3),
                "text": seg_text,
                "words": words_data,
                "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
            }

            all_segments.append(seg_dict)
            last_committed_end = global_end

            if seg_text:
                chunk_text_parts.append(seg_text)

            if on_segment:
                try:
                    on_segment(seg_dict)
                except Exception:
                    pass

        if chunk_text_parts:
            full_text_list.extend(chunk_text_parts)

        current_full_text = " ".join(full_text_list)
        if on_chunk_completed:
            try:
                on_chunk_completed(chunk_idx + 1, total_chunks, all_segments, current_full_text)
            except Exception as save_err:
                _notify(progress_callback, f"Peringatan autosave: {save_err}")

        del chunk_audio
        del raw_segments
        gc.collect()

    return {
        "text": " ".join(full_text_list),
        "segments": all_segments,
        "language": detected_language,
    }


# ---------------------------------------------------------------------------
# 6. PIPELINE HYBRID YOUTUBE (Jalur 1: CC -> Jalur 2: yt-dlp + Whisper)
# ---------------------------------------------------------------------------

def _is_missing_transcript_error(exc: Exception) -> bool:
    """Mendeteksi exception ketidaktersediaan CC/subtitle dari youtube-transcript-api."""
    name = type(exc).__name__
    if name in ("TranscriptsDisabled", "NoTranscriptFound", "VideoUnavailable", "CouldNotRetrieveTranscript"):
        return True
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in ("transcripts are disabled", "no transcripts were found", "could not retrieve a transcript")
    )


def process_youtube(
    url: str,
    use_whisper: bool = False,
    model_name: str = "base",
    language: Optional[str] = "id",
    task: str = "transcribe",
    device: str = "auto",
    force_fp32: bool = False,
    chunk_size_sec: float = 600.0,
    initial_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    on_chunk_completed: Optional[Callable[[int, int, List[Dict[str, Any]], str], None]] = None,
    on_segment: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk_start: Optional[Callable[[int, int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Orkestrasi transkripsi YouTube Hybrid 2 Jalur:
      - Jalur 1 (Default): Ambil CC/subtitle bawaan via youtube-transcript-api.
      - Jalur 2 (Fallback / use_whisper=True): Unduh audio terbaik via yt-dlp ke
        direktori temporary, jalankan pipeline Whisper lokal, lalu bersihkan file temp.
    """
    video_id = extract_video_id(url)

    # Susun prioritas bahasa untuk CC YouTube
    cc_languages: List[str] = []
    if language and language.lower() not in ("auto", "none", ""):
        cc_languages.append(language.lower())
    for fallback_lang in ("id", "en"):
        if fallback_lang not in cc_languages:
            cc_languages.append(fallback_lang)

    # ---- JALUR 1: YouTube CC ----
    if not use_whisper and task != "translate":
        _notify(progress_callback, f"Jalur 1: Mengambil CC/subtitle YouTube untuk video '{video_id}'...")
        try:
            segments = fetch_youtube_transcript(video_id, languages=cc_languages)
            if segments:
                for seg in segments:
                    if on_segment:
                        try:
                            on_segment(seg)
                        except Exception:
                            pass
                full_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
                _notify(
                    progress_callback,
                    f"Jalur 1 berhasil: {len(segments)} segmen CC YouTube diambil.",
                )
                return {
                    "text": full_text,
                    "segments": segments,
                    "language": cc_languages[0] if cc_languages else "auto",
                    "source": "youtube_cc",
                    "video_id": video_id,
                    "duration_str": format_duration(segments[-1]["end"]) if segments else "Tidak diketahui",
                }
            _notify(progress_callback, "CC YouTube kosong. Beralih ke Jalur 2 (yt-dlp + Whisper)...")
        except Exception as exc:
            if _is_missing_transcript_error(exc):
                _notify(
                    progress_callback,
                    f"CC YouTube tidak tersedia ({type(exc).__name__}). Beralih ke Jalur 2 (yt-dlp + Whisper)...",
                )
            else:
                _notify(
                    progress_callback,
                    f"Gagal mengambil CC YouTube ({exc}). Beralih ke Jalur 2 (yt-dlp + Whisper)...",
                )
    elif use_whisper:
        _notify(progress_callback, "Mode Force Whisper aktif. Melewati Jalur 1, langsung ke Jalur 2 (yt-dlp + Whisper)...")
    elif task == "translate":
        _notify(progress_callback, "Mode translate aktif. Menggunakan Jalur 2 (yt-dlp + Whisper)...")

    # ---- JALUR 2: Download Audio via yt-dlp + Whisper ----
    temp_dir = tempfile.mkdtemp(prefix="transkriptor_yt_")
    try:
        _notify(progress_callback, f"Jalur 2: Mengunduh trek audio terbaik YouTube ({video_id})...")
        audio_path = Path(download_youtube_audio(url, temp_dir))

        media_info = probe_media_file(audio_path)
        total_duration = media_info.get("duration") or 0.0

        target_device, fp16, device_desc = detect_optimal_device(
            requested_device=device,
            force_fp32=force_fp32,
            progress_callback=progress_callback,
        )
        _notify(progress_callback, f"Perangkat aktif: {device_desc}")

        model = load_whisper_model(
            model_name=model_name,
            device=target_device,
            force_fp32=force_fp32,
            progress_callback=progress_callback,
        )

        result = transcribe_media_chunked(
            model=model,
            file_path=audio_path,
            total_duration=total_duration,
            chunk_size_sec=chunk_size_sec,
            language=language,
            task=task,
            verbose=False,
            fp16=fp16,
            initial_prompt=initial_prompt,
            on_chunk_completed=on_chunk_completed,
            on_segment=on_segment,
            on_chunk_start=on_chunk_start,
            stop_check=stop_check,
            progress_callback=progress_callback,
        )
        result["source"] = "youtube_whisper"
        result["video_id"] = video_id
        result["duration_str"] = media_info.get("duration_str", "Tidak diketahui")
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
