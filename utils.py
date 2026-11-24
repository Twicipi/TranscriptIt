"""
Module: utils.py
Deskripsi: Layer Model & External I/O Helpers (MVC).
           Menangani verifikasi environment, pemeriksaan media audio/video,
           pengambilan data YouTube (URL parsing, CC fetcher, audio downloader),
           serta pengeksporan hasil transkripsi ke format .txt dan .srt.

Catatan Arsitektur:
  - Bebas dari print(), input(), sys.exit(), maupun referensi widget GUI.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


# ==============================================================================
# 1. PEMERIKSAAN KETERSEDIAAN DEPENDENSI SISTEM
# ==============================================================================

def check_ffmpeg_installed() -> Tuple[bool, Optional[str]]:
    """
    Memeriksa apakah executable FFmpeg dan FFprobe terdeteksi pada PATH sistem.
    Returns:
        Tuple (is_installed, error_message_or_None)
    """
    script_dir = str(Path(__file__).resolve().parent)
    fdm_dir = r"C:\Program Files\Softdeluxe\Free Download Manager"
    for candidate in [script_dir, fdm_dir]:
        if os.path.isdir(candidate) and (Path(candidate) / "ffmpeg.exe").is_file():
            if candidate not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{candidate};{os.environ.get('PATH', '')}"

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        msg = (
            "ERROR: 'ffmpeg' tidak ditemukan di system PATH!\n"
            "faster-whisper membutuhkan FFmpeg untuk membaca dan memproses file audio/video.\n\n"
            "Cara instalasi FFmpeg:\n"
            "- Windows (via winget): winget install Gyan.FFmpeg\n"
            "- Windows (via scoop) : scoop install ffmpeg\n"
            "- Windows (manual)    : Download dari https://ffmpeg.org/download.html lalu tambahkan ke Environment PATH.\n"
            "- Linux (Ubuntu/Debian): sudo apt update && sudo apt install -y ffmpeg\n"
            "- macOS (Homebrew)    : brew install ffmpeg"
        )
        return False, msg

    if not ffprobe_path:
        return True, "Peringatan: 'ffprobe' tidak terdeteksi, beberapa fitur analisis durasi mungkin terbatas."

    return True, None


# ==============================================================================
# 2. FORMATTING WAKTU & UKURAN FILE
# ==============================================================================

def format_file_size(size_bytes: int) -> str:
    """Mengonversi ukuran bytes menjadi format mudah dibaca (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    """Mengonversi detik ke format HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "Tidak diketahui"
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """
    Format detik ke format waktu SubRip (.srt):
    HH:MM:SS,mmm (koma sebagai pemisah milidetik).
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ==============================================================================
# 3. PROBE & ANALISIS METADATA FILE MEDIA
# ==============================================================================

def probe_media_file(file_path: Path) -> Dict[str, Any]:
    """
    Mengekstrak informasi metadata file audio/video (durasi, codec, stream audio)
    menggunakan ffmpeg-python atau langsung via ffprobe CLI.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    file_size = file_path.stat().st_size
    info: Dict[str, Any] = {
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "file_size_str": format_file_size(file_size),
        "duration": None,
        "duration_str": "Tidak diketahui",
        "has_audio": False,
        "audio_codec": None,
        "sample_rate": None,
        "channels": None,
        "raw_probe": None,
    }

    probe_data = None

    try:
        import ffmpeg
        probe_data = ffmpeg.probe(str(file_path))
    except (ImportError, Exception):
        if shutil.which("ffprobe"):
            try:
                cmd = [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    str(file_path),
                ]
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace",
                )
                if proc.returncode == 0 and proc.stdout:
                    probe_data = json.loads(proc.stdout)
            except Exception:
                pass

    if probe_data:
        info["raw_probe"] = probe_data
        format_info = probe_data.get("format", {})

        if "duration" in format_info:
            try:
                dur = float(format_info["duration"])
                info["duration"] = dur
                info["duration_str"] = format_duration(dur)
            except ValueError:
                pass

        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                info["has_audio"] = True
                info["audio_codec"] = stream.get("codec_name", "unknown")
                info["sample_rate"] = stream.get("sample_rate", "unknown")
                info["channels"] = stream.get("channels", "unknown")
                if info["duration"] is None and "duration" in stream:
                    try:
                        dur = float(stream["duration"])
                        info["duration"] = dur
                        info["duration_str"] = format_duration(dur)
                    except ValueError:
                        pass
                break

    return info


# ==============================================================================
# 4. YOUTUBE URL PARSING, CC FETCHER & AUDIO DOWNLOADER
# ==============================================================================

_YOUTUBE_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
})


def is_youtube_url(source: str) -> bool:
    """Memeriksa apakah string input merupakan URL YouTube yang valid."""
    if not source or not isinstance(source, str):
        return False
    raw = source.strip()
    if not raw.lower().startswith(("http://", "https://")):
        return False
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if host not in _YOUTUBE_HOSTS:
            return False
        extract_video_id(raw)
        return True
    except ValueError:
        return False


def extract_video_id(url: str) -> str:
    """
    Mengekstrak YouTube video ID (11 karakter) dari berbagai format URL:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/live/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL YouTube tidak boleh kosong.")

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise ValueError(f"Bukan URL YouTube yang valid: {url}")

    video_id = ""
    path_parts = [p for p in parsed.path.split("/") if p]

    if "youtu.be" in host:
        if path_parts:
            video_id = path_parts[0]
    else:
        if parsed.path == "/watch" or "v" in parse_qs(parsed.query):
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(path_parts) >= 2 and path_parts[0] in ("shorts", "live", "embed", "v"):
            video_id = path_parts[1]

    video_id = video_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Video ID YouTube tidak ditemukan atau tidak valid pada URL: {url}")

    return video_id


def fetch_youtube_transcript(
    video_id: str,
    languages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Mengambil subtitle/CC bawaan YouTube menggunakan youtube-transcript-api
    dan menormalisasinya ke kontrak segmen standar: {"start", "end", "text"}.
    Mendukung youtube-transcript-api versi 0.6.x maupun >=1.0.0.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise ImportError(
            "Library 'youtube-transcript-api' belum terpasang. "
            "Jalankan: pip install youtube-transcript-api"
        ) from exc

    target_langs = languages or ["id", "en"]

    # Kompatibilitas API v0.6.x (classmethod) dan v1.x (instance method)
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        raw_items = YouTubeTranscriptApi.get_transcript(video_id, languages=target_langs)
    else:
        api = YouTubeTranscriptApi()
        raw_items = api.fetch(video_id, languages=target_langs)

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
            text = str(item.get("text", "")).strip()
        else:
            start = float(getattr(item, "start", 0.0))
            duration = float(getattr(item, "duration", 0.0))
            text = str(getattr(item, "text", "")).strip()

        if not text:
            continue

        # Bersihkan newline internal pada CC YouTube agar seragam
        clean_text = re.sub(r"\s+", " ", text).strip()
        normalized.append({
            "start": round(start, 3),
            "end": round(start + max(duration, 0.0), 3),
            "text": clean_text,
        })

    return normalized


def download_youtube_audio(url: str, output_dir: str) -> str:
    """
    Mengunduh trek audio terbaik dari URL YouTube menggunakan yt-dlp ke output_dir.
    Returns:
        Path string menuju file audio yang terunduh.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "Library 'yt-dlp' belum terpasang. Jalankan: pip install yt-dlp"
        ) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_path / "%(id)s.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))
        if not downloaded_path.is_file():
            video_id = info.get("id", "")
            matches = list(out_path.glob(f"{video_id}.*"))
            if matches:
                downloaded_path = matches[0]
            else:
                raise FileNotFoundError(
                    f"File audio hasil unduhan yt-dlp tidak ditemukan di: {output_dir}"
                )
        return str(downloaded_path)


# ==============================================================================
# 5. EXPORT SRT — YouTube-style word-level subtitle
# ==============================================================================

_SENTENCE_END = frozenset(".?!…")
_MAX_CHARS_PER_LINE = 42
_MAX_LINES_PER_BLOCK = 2
_WORD_GAP_SPLIT_SEC = 0.75


def _build_subtitle_blocks(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Membangun blok-blok subtitle setara YouTube dari daftar segmen yang mengandung
    data 'words' (word-level timestamps).
    """
    blocks: List[Dict[str, Any]] = []

    all_words: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words", [])
        if words:
            all_words.extend(words)
        else:
            all_words.append({
                "word": " " + seg.get("text", "").strip(),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            })

    if not all_words:
        return blocks

    block_start: float = all_words[0]["start"]
    block_end: float = all_words[0]["end"]
    lines: List[str] = [""]
    line_idx: int = 0

    def _flush_block() -> None:
        nonlocal block_start, block_end, lines, line_idx
        text = "\n".join(line.strip() for line in lines if line.strip())
        if text:
            blocks.append({
                "start": block_start,
                "end": block_end,
                "text": text,
            })
        lines = [""]
        line_idx = 0

    for i, w in enumerate(all_words):
        word_text = w.get("word", "")
        w_start = w.get("start", 0.0)
        w_end = w.get("end", w_start)

        if i == 0:
            block_start = w_start
            block_end = w_end
            lines[0] = word_text.lstrip()
            continue

        prev_end = all_words[i - 1].get("end", 0.0)
        gap = w_start - prev_end
        word_stripped = word_text.lstrip()

        proposed = (lines[line_idx] + word_text).rstrip()
        proposed_len = len(proposed)

        trigger_new_block = False
        trigger_new_line = False

        if gap > _WORD_GAP_SPLIT_SEC:
            trigger_new_block = True
        elif lines[line_idx].rstrip() and lines[line_idx].rstrip()[-1] in _SENTENCE_END:
            trigger_new_block = True
        elif proposed_len > _MAX_CHARS_PER_LINE:
            if line_idx < _MAX_LINES_PER_BLOCK - 1:
                trigger_new_line = True
            else:
                trigger_new_block = True

        if trigger_new_block:
            block_end = all_words[i - 1].get("end", block_end)
            _flush_block()
            block_start = w_start
            lines[0] = word_stripped
            block_end = w_end
        elif trigger_new_line:
            lines.append(word_stripped)
            line_idx += 1
            block_end = w_end
        else:
            lines[line_idx] = proposed
            block_end = w_end

    if any(line.strip() for line in lines):
        block_end = all_words[-1].get("end", block_end)
        _flush_block()

    return blocks


def export_srt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Menyimpan subtitle ke file .srt secara polimorfik dari sumber lokal maupun YouTube.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_words = any(seg.get("words") for seg in segments)

    if has_words:
        blocks = _build_subtitle_blocks(segments)
    else:
        blocks = [
            {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, start=1):
            text = block.get("text", "").strip()
            if not text or not re.sub(r"[\s.,]+", "", text):
                continue
            start_ts = format_timestamp_srt(block["start"])
            end_ts = format_timestamp_srt(block["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")

    return output_path


# ==============================================================================
# 6. EXPORT TXT — Paragraf alami dengan deteksi jeda hening
# ==============================================================================

_PARAGRAPH_GAP_SEC = 2.5


def export_txt(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    include_timestamps: bool = True,
    file_name: str = "",
    duration_str: str = "",
) -> Path:
    """
    Menyimpan teks transkripsi ke .txt secara polimorfik dengan paragraf alami.
    """
    if output_path is None:
        raise ValueError("output_path harus ditentukan.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if file_name or duration_str:
            f.write("=" * 70 + "\n")
            f.write(f"  HASIL TRANSKRIPSI: {file_name}\n")
            if duration_str:
                f.write(f"  Durasi Total: {duration_str}\n")
            f.write("=" * 70 + "\n\n")

        if segments:
            if include_timestamps:
                f.write("--- [TRANSKRIP BERSTEMPEL WAKTU] ---\n\n")

            clean_segments: List[Dict[str, Any]] = []
            for seg in segments:
                seg_text = seg.get("text", "").strip()
                if not re.sub(r"[\s.,]+", "", seg_text):
                    continue
                clean_segments.append(seg)
                if include_timestamps:
                    start_ts = format_duration(seg.get("start", 0.0))
                    f.write(f"[{start_ts}] {seg_text}\n")

            if clean_segments:
                if include_timestamps:
                    f.write("\n" + "=" * 70 + "\n")
                f.write("--- [TRANSKRIP TEKS LENGKAP] ---\n\n")

                paragraph_parts: List[str] = []
                prev_end: Optional[float] = None

                for seg in clean_segments:
                    seg_start = seg.get("start", 0.0)
                    seg_text = seg.get("text", "").strip()

                    if prev_end is not None and (seg_start - prev_end) > _PARAGRAPH_GAP_SEC:
                        if paragraph_parts:
                            f.write(" ".join(paragraph_parts) + "\n\n")
                            paragraph_parts = []

                    paragraph_parts.append(seg_text)
                    prev_end = seg.get("end", seg_start)

                if paragraph_parts:
                    f.write(" ".join(paragraph_parts) + "\n")
        else:
            f.write(text.strip() + "\n")

    return output_path
