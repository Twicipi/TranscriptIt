"""
Module: utils.py
Deskripsi: Utilitas pendukung untuk verifikasi environment, pemeriksaan media audio/video,
           dan pengeksporan hasil transkripsi ke format .txt dan .srt setara kualitas YouTube.

Perubahan v2 (faster-whisper refactor):
  - export_srt()  : Gunakan word-level timestamps untuk subtitle YouTube-style
                    (maks 42 karakter/baris, 2 baris/blok, pecah pada jeda > 0.75 detik
                     atau tanda baca akhir kalimat).
  - export_txt()  : Gabungkan segmen berdekatan menjadi paragraf alami; paragraf baru
                    hanya ketika jeda antar segmen > 2.5 detik.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pastikan output terminal mendukung UTF-8 di Windows
if sys.platform == "win32":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        secs  += 1
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
        "file_name":      file_path.name,
        "file_size_bytes": file_size,
        "file_size_str":  format_file_size(file_size),
        "duration":       None,
        "duration_str":   "Tidak diketahui",
        "has_audio":      False,
        "audio_codec":    None,
        "sample_rate":    None,
        "channels":       None,
        "raw_probe":      None,
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
                info["has_audio"]   = True
                info["audio_codec"] = stream.get("codec_name", "unknown")
                info["sample_rate"] = stream.get("sample_rate", "unknown")
                info["channels"]    = stream.get("channels", "unknown")
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
# 4. EXPORT SRT — YouTube-style word-level subtitle
# ==============================================================================

# Karakter kalimat akhir yang memicu pemecahan blok subtitle
_SENTENCE_END = frozenset(".?!…")
# Pemisah baris dalam satu blok subtitle
_MAX_CHARS_PER_LINE  = 42
_MAX_LINES_PER_BLOCK = 2
# Jeda antar kata (detik) yang memaksa pemecahan blok baru
_WORD_GAP_SPLIT_SEC  = 0.75


def _build_subtitle_blocks(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Membangun blok-blok subtitle setara YouTube dari daftar segmen yang mengandung
    data 'words' (word-level timestamps).

    Strategi:
      1. Kumpulkan semua kata dari seluruh segmen secara berurutan.
      2. Pecah blok baru ketika:
         a. Jeda antar kata > _WORD_GAP_SPLIT_SEC detik.
         b. Kata berikutnya tidak muat di baris saat ini (maks 42 karakter).
         c. Baris saat ini sudah penuh dan blok sudah mencapai 2 baris.
         d. Kata sebelumnya diakhiri tanda baca kalimat (. ? ! …).
      3. Fallback ke timestamp segmen jika 'words' kosong.
    """
    blocks: List[Dict[str, Any]] = []

    # Kumpulkan semua kata dari semua segmen
    all_words: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words", [])
        if words:
            all_words.extend(words)
        else:
            # Fallback: perlakukan seluruh segmen sebagai satu "kata"
            all_words.append({
                "word":  " " + seg.get("text", "").strip(),
                "start": seg.get("start", 0.0),
                "end":   seg.get("end", 0.0),
            })

    if not all_words:
        return blocks

    # State builder
    block_start: float = all_words[0]["start"]
    block_end: float   = all_words[0]["end"]
    lines: List[str]   = [""]   # Daftar baris dalam blok saat ini
    line_idx: int       = 0

    def _flush_block() -> None:
        nonlocal block_start, block_end, lines, line_idx
        text = "\n".join(l.strip() for l in lines if l.strip())
        if text:
            blocks.append({
                "start": block_start,
                "end":   block_end,
                "text":  text,
            })
        lines   = [""]
        line_idx = 0

    for i, w in enumerate(all_words):
        word_text = w.get("word", "")
        w_start   = w.get("start", 0.0)
        w_end     = w.get("end", w_start)

        if i == 0:
            block_start = w_start
            block_end   = w_end
            lines[0]    = word_text.lstrip()
            continue

        prev_end = all_words[i - 1].get("end", 0.0)
        gap      = w_start - prev_end
        word_stripped = word_text.lstrip()

        # Hitung apakah kata ini muat di baris saat ini
        proposed = (lines[line_idx] + word_text).rstrip()
        proposed_len = len(proposed)

        # Periksa pemicu pecah blok baru
        trigger_new_block = False
        trigger_new_line  = False

        # a) Jeda antar kata terlalu besar
        if gap > _WORD_GAP_SPLIT_SEC:
            trigger_new_block = True

        # b) Kata sebelumnya diakhiri tanda baca kalimat
        elif lines[line_idx].rstrip() and lines[line_idx].rstrip()[-1] in _SENTENCE_END:
            trigger_new_block = True

        # c) Kata tidak muat di baris saat ini
        elif proposed_len > _MAX_CHARS_PER_LINE:
            # Coba pindah ke baris berikutnya dalam blok yang sama
            if line_idx < _MAX_LINES_PER_BLOCK - 1:
                trigger_new_line = True
            else:
                # Blok sudah penuh → buat blok baru
                trigger_new_block = True

        if trigger_new_block:
            block_end = all_words[i - 1].get("end", block_end)
            _flush_block()
            block_start = w_start
            lines[0]    = word_stripped
            block_end   = w_end
        elif trigger_new_line:
            lines.append(word_stripped)
            line_idx += 1
            block_end = w_end
        else:
            lines[line_idx] = proposed
            block_end = w_end

    # Flush sisa blok terakhir
    if any(l.strip() for l in lines):
        block_end = all_words[-1].get("end", block_end)
        _flush_block()

    return blocks


def export_srt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Menyimpan subtitle ke file .srt setara kualitas YouTube.

    Jika segmen mengandung data 'words' (word-level timestamps dari faster-whisper),
    gunakan algoritma _build_subtitle_blocks() untuk membentuk blok yang:
      - Maks 42 karakter per baris
      - Maks 2 baris per blok
      - Pecah pada jeda > 0.75 detik atau tanda baca akhir kalimat
    
    Jika tidak ada data 'words' (fallback kompatibilitas), gunakan teks segmen langsung.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cek apakah ada data word-level timestamps
    has_words = any(seg.get("words") for seg in segments)

    if has_words:
        blocks = _build_subtitle_blocks(segments)
    else:
        # Fallback: satu segmen = satu blok subtitle (perilaku lama)
        blocks = [
            {
                "start": seg.get("start", 0.0),
                "end":   seg.get("end", 0.0),
                "text":  seg.get("text", "").strip(),
            }
            for seg in segments
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, start=1):
            text = block.get("text", "").strip()
            if not text:
                continue
            # Filter blok yang hanya berisi noise (titik/spasi berulang)
            if not re.sub(r"[\s.,]+", "", text):
                continue
            start_ts = format_timestamp_srt(block["start"])
            end_ts   = format_timestamp_srt(block["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")

    return output_path


# ==============================================================================
# 5. EXPORT TXT — Paragraf alami dengan deteksi jeda hening
# ==============================================================================

_PARAGRAPH_GAP_SEC = 2.5   # Jeda antar segmen (detik) yang memicu paragraf baru


def export_txt(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    include_timestamps: bool = True,
    file_name: str = "",
    duration_str: str = "",
) -> Path:
    """
    Menyimpan teks transkripsi ke .txt yang rapi dan mudah dibaca.

    - Bagian berstempel waktu: setiap segmen pada baris baru dengan format [HH:MM:SS].
    - Bagian teks lengkap: segmen-segmen digabungkan menjadi paragraf alami.
      Paragraf baru dibuat hanya ketika jeda antar segmen > 2.5 detik.
    """
    if output_path is None:
        raise ValueError("output_path harus ditentukan.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        if file_name or duration_str:
            f.write("=" * 70 + "\n")
            f.write(f"  HASIL TRANSKRIPSI: {file_name}\n")
            if duration_str:
                f.write(f"  Durasi Total: {duration_str}\n")
            f.write("=" * 70 + "\n\n")

        if segments:
            # --- Bagian berstempel waktu ---
            f.write("--- [TRANSKRIP BERSTEMPEL WAKTU] ---\n\n")

            clean_segments: List[Dict[str, Any]] = []
            for seg in segments:
                seg_text = seg.get("text", "").strip()
                if not re.sub(r"[\s.,]+", "", seg_text):
                    continue
                clean_segments.append(seg)
                start_ts = format_duration(seg.get("start", 0.0))
                f.write(f"[{start_ts}] {seg_text}\n")

            # --- Bagian teks lengkap dengan paragraf alami ---
            if clean_segments:
                f.write("\n" + "=" * 70 + "\n")
                f.write("--- [TRANSKRIP TEKS LENGKAP] ---\n\n")

                paragraph_parts: List[str] = []
                prev_end: Optional[float] = None

                for seg in clean_segments:
                    seg_start = seg.get("start", 0.0)
                    seg_text  = seg.get("text", "").strip()

                    # Jeda hening > _PARAGRAPH_GAP_SEC → mulai paragraf baru
                    if prev_end is not None and (seg_start - prev_end) > _PARAGRAPH_GAP_SEC:
                        if paragraph_parts:
                            f.write(" ".join(paragraph_parts) + "\n\n")
                            paragraph_parts = []

                    paragraph_parts.append(seg_text)
                    prev_end = seg.get("end", seg_start)

                # Tulis sisa paragraf terakhir
                if paragraph_parts:
                    f.write(" ".join(paragraph_parts) + "\n")
        else:
            # Fallback jika tidak ada segmen
            f.write(text.strip() + "\n")

    return output_path


# ==============================================================================
# 6. CLI UI STYLING & BANNER
# ==============================================================================

class Colors:
    """ANSI color codes untuk output terminal yang estetik dan informatif."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}======================================================================
  🎙️  WHISPER TRANSKRIPTOR CLI — faster-whisper ENGINE v2
  Transkripsi Audio/Video Skala Besar | VAD | Word Timestamps | YouTube-SRT
======================================================================{Colors.RESET}"""
    print(banner)
    sys.stdout.flush()


def print_info(title: str, message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {Colors.BOLD}{title}:{Colors.RESET} {message}")
    sys.stdout.flush()


def print_success(message: str):
    print(f"{Colors.GREEN}{Colors.BOLD}[SUKSES]{Colors.RESET} {message}")
    sys.stdout.flush()


def print_warning(message: str):
    print(f"{Colors.YELLOW}{Colors.BOLD}[PERINGATAN]{Colors.RESET} {message}")
    sys.stdout.flush()


def print_error(message: str):
    print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} {message}", file=sys.stderr)
    sys.stderr.flush()
