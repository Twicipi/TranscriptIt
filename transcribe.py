#!/usr/bin/env python3
"""
CLI Tool: Transkriptor Whisper
Deskripsi: Alat bantu CLI untuk transkripsi file audio/video durasi panjang
           (hingga 8GB / 10 jam) dengan otomatisasi GPU/CPU, output .txt dan .srt.

Penggunaan Contoh:
    python transcribe.py WGTIK_DAY_1.mp4
    python transcribe.py rekaman.m4a --model small --language id
    python transcribe.py film.mkv --model medium --device cuda --output-dir ./hasil
"""

import argparse
import sys
import time
from pathlib import Path

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
    AVAILABLE_MODELS,
    detect_optimal_device,
    load_whisper_model,
    transcribe_media_chunked,
)
from utils import (
    Colors,
    check_ffmpeg_installed,
    export_srt,
    export_txt,
    format_duration,
    format_file_size,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
    probe_media_file,
)

# Format ekstensi media yang umum didukung
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mp3", ".m4a", ".wav",
    ".flac", ".ogg", ".aac", ".webm", ".avi",
    ".mov", ".wmv", ".ts", ".m4v", ".wma", ".opus"
}


def build_parser() -> argparse.ArgumentParser:
    """Membangun argument parser CLI yang ramah pengguna."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="CLI Tool Transkripsi Audio & Video Skala Besar menggunakan faster-whisper (CTranslate2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  1. Transkripsi standar (Bahasa Indonesia, model base, auto GPU/CPU):
     python transcribe.py podcast.mp3

  2. Menggunakan model yang lebih akurat (small/medium) dan simpan ke folder tertentu:
     python transcribe.py kuliah_3jam.mp4 --model small --output-dir ./hasil_transkrip

  3. Menampilkan teks langsung saat transkripsi berjalan (verbose):
     python transcribe.py seminar.mkv --model small --verbose

  4. Deteksi bahasa otomatis (jika ada bahasa campuran / Inggris):
     python transcribe.py conference.mp4 --language auto

  5. Memaksa proses di CPU (misal GPU VRAM terbatas):
     python transcribe.py video_10jam.mp4 --model base --device cpu
        """,
    )

    # Argumen Input File (bisa posisi langsung atau via flag -i/--input)
    parser.add_argument(
        "file",
        nargs="?",
        type=str,
        default=None,
        help="Path ke file media audio/video (contoh: rekaman.mp3, video.mp4, dll).",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_flag",
        type=str,
        default=None,
        help="Alternatif path ke file media input.",
    )

    # Opsi Model
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="base",
        help="Pilihan model Whisper: tiny, base, small, medium, large-v3-turbo (rekomendasi GPU>=4GB), large-v3. Default: 'base'.",
    )

    # Opsi Bahasa
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="id",
        help="Kode bahasa audio (default: 'id' untuk Bahasa Indonesia, gunakan 'auto' untuk deteksi otomatis).",
    )

    # Opsi Output
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Direktori penyimpanan hasil transkrip (.txt dan .srt). Default: sama dengan folder file input.",
    )

    # Opsi Hardware Perangkat
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Pilihan perangkat komputasi: 'auto' (prioritas GPU CUDA jika ada), 'cuda', atau 'cpu'. Default: 'auto'.",
    )

    # Mode Verbose / Progress
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Tampilkan streaming log teks per segmen secara langsung saat transkripsi (default: True).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sembunyikan teks streaming, hanya tampilkan progress bagian per bagian.",
    )

    # Ukuran Chunk (dalam menit)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Durasi tiap bagian chunk dalam menit (default: 10 menit). Mencegah kehabisan memori untuk file besar.",
    )

    # Opsi Tambahan
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Paksa penggunaan FP32 (Full Precision) di GPU untuk mencegah ketidakstabilan numerik.",
    )
    parser.add_argument(
        "--initial-prompt",
        "-p",
        type=str,
        default=None,
        help="Teks petunjuk awal/kosakata khusus (misal nama tokoh, istilah teknis) untuk memandu akurasi Whisper.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Mode tugas: 'transcribe' (transkrip sesuai bahasa asli) atau 'translate' (terjemahkan ke Bahasa Inggris).",
    )

    return parser


def main() -> int:
    """Fungsi utama eksekusi CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Cetak header banner
    print_banner()

    # 1. Tentukan path file input
    input_path_str = args.file or args.input_flag
    if not input_path_str:
        print_error("Tidak ada file input yang ditentukan!")
        print(f"Jalankan {Colors.BOLD}python transcribe.py -h{Colors.RESET} untuk melihat panduan lengkap.\n")
        return 1

    input_file = Path(input_path_str).resolve()
    if not input_file.exists():
        print_error(f"File tidak ditemukan pada path: {input_file}")
        return 1

    # 2. Cek dependensi sistem (FFmpeg)
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
    if not ffmpeg_ok:
        print_error(ffmpeg_msg)
        return 1
    elif ffmpeg_msg:
        print_warning(ffmpeg_msg)

    # 3. Validasi ekstensi file
    ext = input_file.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print_warning(
            f"Ekstensi file '{ext}' tidak umum untuk audio/video. "
            f"Akan tetap dicoba diproses melalui FFmpeg."
        )

    # 4. Probe informasi media
    print_info("Media Inspection", f"Memeriksa file: {Colors.BOLD}{input_file.name}{Colors.RESET}")
    media_info = probe_media_file(input_file)
    
    if not media_info.get("has_audio", True) and media_info.get("raw_probe") is not None:
        print_warning(
            "Tidak ditemukan stream audio pada file ini! "
            "Pastikan file memiliki trek audio aktif."
        )

    # 5. Deteksi perangkat & presisi (CUDA vs CPU)
    device, fp16, device_desc = detect_optimal_device(
        requested_device=args.device,
        force_fp32=args.force_fp32
    )

    # 6. Tentukan direktori output
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    stem_name = input_file.stem
    txt_path = output_dir / f"{stem_name}.txt"
    srt_path = output_dir / f"{stem_name}.srt"

    verbose_mode = not args.quiet

    # Tampilkan ringkasan konfigurasi sebelum eksekusi
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

    # 7. Muat Model Whisper
    try:
        model = load_whisper_model(
            model_name=args.model,
            device=device,
            force_fp32=args.force_fp32,
        )
    except Exception as e:
        print_error(f"Gagal memuat model Whisper: {e}")
        return 1

    # 8. Eksekusi Transkripsi dengan Smart Chunking
    print(f"{Colors.GREEN}{Colors.BOLD}🚀 Memulai proses transkripsi (Autosave aktif per chunk)...{Colors.RESET}\n")
    sys.stdout.flush()

    start_transcribe = time.time()

    def on_chunk_done(chunk_idx: int, total_chunks: int, all_segs: list, full_txt: str):
        # Langsung tulis ke disk setiap kali 1 chunk selesai
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

    # Finalisasi penyimpanan
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

    # 10. Laporan Akhir
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


if __name__ == "__main__":
    sys.exit(main())
