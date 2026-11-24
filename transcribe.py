#!/usr/bin/env python3
"""
CLI Tool: Transkriptor Whisper (Controller & View)
Deskripsi: Alat bantu CLI untuk transkripsi file audio/video lokal durasi panjang
           maupun URL YouTube (Hybrid: CC cepat atau fallback yt-dlp + Whisper).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Pastikan encoding console terminal selalu UTF-8 di Windows
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from transcriber import (
    detect_optimal_device,
    load_whisper_model,
    process_youtube,
    transcribe_media_chunked,
)
from utils import (
    check_ffmpeg_installed,
    export_srt,
    export_txt,
    extract_video_id,
    format_duration,
    is_youtube_url,
    probe_media_file,
)

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mp3", ".m4a", ".wav",
    ".flac", ".ogg", ".aac", ".webm", ".avi",
    ".mov", ".wmv", ".ts", ".m4v", ".wma", ".opus",
}


# ==============================================================================
# CLI VIEW HELPERS (MVC: View hanya berada di layer Controller/View)
# ==============================================================================

class Colors:
    """ANSI color codes untuk output terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"


def print_banner() -> None:
    banner = f"""
{Colors.CYAN}{Colors.BOLD}======================================================================
  🎙️  WHISPER TRANSKRIPTOR CLI — faster-whisper ENGINE v2
  Transkripsi Audio/Video & YouTube | VAD | Word Timestamps | YouTube-SRT
======================================================================{Colors.RESET}"""
    print(banner)
    sys.stdout.flush()


def print_info(title: str, message: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {Colors.BOLD}{title}:{Colors.RESET} {message}")
    sys.stdout.flush()


def print_success(message: str) -> None:
    print(f"{Colors.GREEN}{Colors.BOLD}[SUKSES]{Colors.RESET} {message}")
    sys.stdout.flush()


def print_warning(message: str) -> None:
    print(f"{Colors.YELLOW}{Colors.BOLD}[PERINGATAN]{Colors.RESET} {message}")
    sys.stdout.flush()


def print_error(message: str) -> None:
    print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} {message}", file=sys.stderr)
    sys.stderr.flush()


# ==============================================================================
# CLI ARGUMENT PARSER & CONTROLLER
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Membangun argument parser CLI yang ramah pengguna."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="CLI Tool Transkripsi Audio/Video & YouTube menggunakan faster-whisper (CTranslate2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  1. Transkripsi file lokal (Bahasa Indonesia, model base, auto GPU/CPU):
     python transcribe.py podcast.mp3

  2. Transkripsi dari YouTube URL (Jalur 1: ambil CC bawaan cepat):
     python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

  3. Paksa transkripsi YouTube via yt-dlp + Whisper (abaikan CC bawaan):
     python transcribe.py "https://youtu.be/dQw4w9WgXcQ" --force-whisper --model small

  4. Menggunakan model yang lebih akurat dan simpan ke folder tertentu:
     python transcribe.py kuliah_3jam.mp4 --model large-v3-turbo --output-dir ./hasil_transkrip
        """,
    )

    parser.add_argument(
        "file",
        nargs="?",
        type=str,
        default=None,
        help="Path ke file media audio/video lokal ATAU URL YouTube.",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_flag",
        type=str,
        default=None,
        help="Alternatif path file media lokal atau URL YouTube.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="base",
        help="Pilihan model Whisper: tiny, base, small, medium, large-v3-turbo, large-v3. Default: 'base'.",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="id",
        help="Kode bahasa audio (default: 'id', gunakan 'auto' untuk deteksi otomatis).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Direktori penyimpanan hasil transkrip (.txt dan .srt). Default: folder file input atau direktori kerja saat ini.",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Pilihan perangkat komputasi: 'auto', 'cuda', atau 'cpu'. Default: 'auto'.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Tampilkan streaming log teks per segmen secara langsung saat transkripsi (default: True).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sembunyikan teks streaming, hanya tampilkan status ringkas.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Durasi tiap bagian chunk dalam menit (default: 10 menit).",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Paksa penggunaan FP32 (Full Precision) di GPU.",
    )
    parser.add_argument(
        "--force-whisper",
        action="store_true",
        help="Paksa unduh audio YouTube (yt-dlp) + transkripsi Whisper meskipun CC YouTube tersedia.",
    )
    parser.add_argument(
        "--initial-prompt",
        "-p",
        type=str,
        default=None,
        help="Teks petunjuk awal/kosakata khusus untuk memandu akurasi Whisper.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Mode tugas: 'transcribe' atau 'translate' (ke Bahasa Inggris).",
    )

    return parser


def _make_segment_printer(verbose_mode: bool):
    def _on_segment(seg: Dict[str, Any]) -> None:
        if verbose_mode:
            t_start = format_duration(seg.get("start", 0.0))
            t_end = format_duration(seg.get("end", 0.0))
            print(f"  {Colors.DIM}[{t_start} --> {t_end}]{Colors.RESET} {seg.get('text', '').strip()}")
            sys.stdout.flush()
    return _on_segment


def _run_youtube_cli(args: argparse.Namespace, url: str) -> int:
    """Menangani alur CLI untuk input URL YouTube."""
    try:
        video_id = extract_video_id(url)
    except ValueError as exc:
        print_error(str(exc))
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"youtube_{video_id}.txt"
    srt_path = output_dir / f"youtube_{video_id}.srt"
    verbose_mode = not args.quiet

    if args.force_whisper:
        ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
        if not ffmpeg_ok:
            print_error(ffmpeg_msg or "FFmpeg tidak ditemukan.")
            return 1
        elif ffmpeg_msg:
            print_warning(ffmpeg_msg)

    print(f"""
{Colors.CYAN}----------------------------------------------------------------------
  📋 RINGKASAN SESI TRANSKRIPSI YOUTUBE:
----------------------------------------------------------------------{Colors.RESET}
  🔗 URL YouTube     : {Colors.BOLD}{url}{Colors.RESET}
  🆔 Video ID        : {video_id}
  🛤️  Mode Jalur      : {'Force Whisper (yt-dlp + Whisper)' if args.force_whisper else 'Hybrid (CC Cepat -> Fallback Whisper)'}
  🧠 Model Whisper   : {args.model}
  🌐 Bahasa          : {args.language.upper() if args.language else 'Auto-detect'}
  📂 Output File     :
     - {txt_path}
     - {srt_path}
{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}
""")
    sys.stdout.flush()

    def cli_progress(msg: str) -> None:
        print_info("YouTube", msg)

    def on_chunk_done(chunk_idx: int, total_chunks: int, all_segs: list, full_txt: str) -> None:
        export_txt(
            text=full_txt,
            segments=all_segs,
            output_path=txt_path,
            include_timestamps=True,
            file_name=f"YouTube ({video_id})",
        )
        export_srt(segments=all_segs, output_path=srt_path)
        print(f"  {Colors.GREEN}💾 [Autosave]{Colors.RESET} Progress bagian {chunk_idx}/{total_chunks} tersimpan.")
        sys.stdout.flush()

    start_time = time.time()
    try:
        result = process_youtube(
            url=url,
            use_whisper=args.force_whisper,
            model_name=args.model,
            language=args.language,
            task=args.task,
            device=args.device,
            force_fp32=args.force_fp32,
            chunk_size_sec=float(args.chunk_size * 60),
            initial_prompt=args.initial_prompt,
            progress_callback=cli_progress,
            on_chunk_completed=on_chunk_done,
            on_segment=_make_segment_printer(verbose_mode),
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[DIBATALKAN]{Colors.RESET} Transkripsi YouTube dihentikan pengguna.")
        return 130
    except Exception as e:
        print_error(f"Gagal memproses YouTube: {e}")
        return 1

    elapsed_time = time.time() - start_time
    segments = result.get("segments", [])
    full_text = result.get("text", "")
    detected_language = result.get("language", args.language)
    source_label = result.get("source", "youtube")
    duration_str = result.get("duration_str", "Tidak diketahui")

    try:
        export_txt(
            text=full_text,
            segments=segments,
            output_path=txt_path,
            include_timestamps=True,
            file_name=f"YouTube ({video_id})",
            duration_str=duration_str,
        )
        export_srt(segments=segments, output_path=srt_path)
    except Exception as e:
        print_error(f"Gagal menyimpan hasil: {e}")
        return 1

    print(f"""
{Colors.GREEN}======================================================================
  🎉 TRANSKRIPSI YOUTUBE SELESAI! ({source_label})
======================================================================{Colors.RESET}
  📄 Dokumen Teks (.txt) : {Colors.BOLD}{txt_path}{Colors.RESET}
  🎬 Subtitle Film (.srt): {Colors.BOLD}{srt_path}{Colors.RESET}
  🔢 Total Segmen        : {len(segments)} segmen
  🌐 Bahasa              : {detected_language}
  ⏱️  Waktu Komputasi    : {format_duration(elapsed_time)} ({elapsed_time:.1f}s)
{Colors.GREEN}======================================================================{Colors.RESET}
""")
    sys.stdout.flush()
    return 0


def _run_local_file_cli(args: argparse.Namespace, input_path_str: str) -> int:
    """Menangani alur CLI untuk input file media lokal."""
    input_file = Path(input_path_str).resolve()
    if not input_file.exists():
        print_error(f"File tidak ditemukan pada path: {input_file}")
        return 1

    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
    if not ffmpeg_ok:
        print_error(ffmpeg_msg or "FFmpeg tidak ditemukan.")
        return 1
    elif ffmpeg_msg:
        print_warning(ffmpeg_msg)

    ext = input_file.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print_warning(
            f"Ekstensi file '{ext}' tidak umum untuk audio/video. "
            f"Akan tetap dicoba diproses melalui FFmpeg."
        )

    print_info("Media Inspection", f"Memeriksa file: {Colors.BOLD}{input_file.name}{Colors.RESET}")
    media_info = probe_media_file(input_file)

    if not media_info.get("has_audio", True) and media_info.get("raw_probe") is not None:
        print_warning(
            "Tidak ditemukan stream audio pada file ini! "
            "Pastikan file memiliki trek audio aktif."
        )

    def cli_progress(msg: str) -> None:
        print_info("Engine", msg)

    device, fp16, device_desc = detect_optimal_device(
        requested_device=args.device,
        force_fp32=args.force_fp32,
        progress_callback=cli_progress,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_name = input_file.stem
    txt_path = output_dir / f"{stem_name}.txt"
    srt_path = output_dir / f"{stem_name}.srt"
    verbose_mode = not args.quiet

    print(f"""
{Colors.CYAN}----------------------------------------------------------------------
  📋 RINGKASAN SESI TRANSKRIPSI:
----------------------------------------------------------------------{Colors.RESET}
  📁 File Target     : {Colors.BOLD}{input_file.name}{Colors.RESET}
  📊 Ukuran File     : {media_info['file_size_str']}
  ⏱️  Estimasi Durasi : {media_info['duration_str']}
  🔊 Audio Codec     : {media_info['audio_codec'] or 'Auto-detect'} ({media_info['sample_rate'] or '-'} Hz, {media_info['channels'] or '-'} ch)
  🧠 Model Whisper   : {Colors.BOLD}{args.model}{Colors.RESET}
  🖥️  Perangkat       : {device_desc}
  🌐 Bahasa          : {args.language.upper() if args.language else 'Auto-detect'}
  🧩 Smart Chunking  : Aktif ({args.chunk_size} menit per bagian)
  📝 Mode Tampilan   : {'Streaming Teks (Verbose)' if verbose_mode else 'Quiet (Ringkas)'}
  📂 Output File     :
     - {txt_path}
     - {srt_path}
{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}
""")
    sys.stdout.flush()

    try:
        model = load_whisper_model(
            model_name=args.model,
            device=device,
            force_fp32=args.force_fp32,
            progress_callback=cli_progress,
        )
    except Exception as e:
        print_error(f"Gagal memuat model Whisper: {e}")
        return 1

    print(f"{Colors.GREEN}{Colors.BOLD}🚀 Memulai proses transkripsi (Autosave aktif per chunk)...{Colors.RESET}\n")
    sys.stdout.flush()

    start_transcribe = time.time()

    def on_chunk_done(chunk_idx: int, total_chunks: int, all_segs: list, full_txt: str) -> None:
        export_txt(
            text=full_txt,
            segments=all_segs,
            output_path=txt_path,
            include_timestamps=True,
            file_name=input_file.name,
            duration_str=media_info.get("duration_str", ""),
        )
        export_srt(segments=all_segs, output_path=srt_path)
        print(f"  {Colors.GREEN}💾 [Autosave]{Colors.RESET} Progress bagian {chunk_idx}/{total_chunks} tersimpan ke '{txt_path.name}' & '{srt_path.name}'")
        sys.stdout.flush()

    try:
        result = transcribe_media_chunked(
            model=model,
            file_path=input_file,
            total_duration=media_info.get("duration") or 0.0,
            chunk_size_sec=float(args.chunk_size * 60),
            language=args.language,
            task=args.task,
            verbose=verbose_mode,
            fp16=fp16,
            initial_prompt=args.initial_prompt,
            on_chunk_completed=on_chunk_done,
            on_segment=_make_segment_printer(verbose_mode),
            progress_callback=cli_progress,
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[DIBATALKAN]{Colors.RESET} Transkripsi dihentikan pengguna. Hasil segmen sebelumnya tetap tersimpan di disk.")
        return 130
    except Exception as e:
        print_error(f"Terjadi kesalahan saat proses transkripsi: {e}")
        return 1

    elapsed_time = time.time() - start_transcribe
    segments = result.get("segments", [])
    full_text = result.get("text", "")
    detected_language = result.get("language", args.language)

    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Transkripsi selesai dalam {format_duration(elapsed_time)} ({elapsed_time:.1f} detik)!{Colors.RESET}")

    try:
        export_txt(
            text=full_text,
            segments=segments,
            output_path=txt_path,
            include_timestamps=True,
            file_name=input_file.name,
            duration_str=media_info.get("duration_str", ""),
        )
        export_srt(segments=segments, output_path=srt_path)
    except Exception as e:
        print_error(f"Gagal finalisasi penyimpanan: {e}")
        return 1

    audio_dur = media_info.get("duration")
    speed_factor_str = ""
    if audio_dur and audio_dur > 0:
        speed_factor = audio_dur / max(elapsed_time, 0.01)
        speed_factor_str = f" ({speed_factor:.1f}x lebih cepat dari durasi asli media)"

    print(f"""
{Colors.GREEN}======================================================================
  🎉 SELURUH TRANSKRIPSI BERHASIL DISELESAIKAN!
======================================================================{Colors.RESET}
  📄 Dokumen Teks (.txt) : {Colors.BOLD}{txt_path}{Colors.RESET}
  🎬 Subtitle Film (.srt): {Colors.BOLD}{srt_path}{Colors.RESET}
  🔢 Total Segmen        : {len(segments)} segmen
  🌐 Bahasa Terdeteksi   : {detected_language}
  ⏱️  Waktu Komputasi    : {format_duration(elapsed_time)}{speed_factor_str}
{Colors.GREEN}======================================================================{Colors.RESET}
""")
    sys.stdout.flush()
    return 0


def main() -> int:
    """Fungsi utama eksekusi CLI."""
    parser = build_parser()
    args = parser.parse_args()

    print_banner()

    input_path_str = args.file or args.input_flag
    if not input_path_str:
        print_error("Tidak ada file input atau URL YouTube yang ditentukan!")
        print(f"Jalankan {Colors.BOLD}python transcribe.py -h{Colors.RESET} untuk melihat panduan lengkap.\n")
        return 1

    if is_youtube_url(input_path_str):
        return _run_youtube_cli(args, input_path_str)
    return _run_local_file_cli(args, input_path_str)


if __name__ == "__main__":
    sys.exit(main())
