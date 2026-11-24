#!/usr/bin/env python3
"""
Whisper Transkriptor Studio - CustomTkinter desktop interface.
"""

import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if (APP_DIR / "ffmpeg.exe").is_file():
    if str(APP_DIR) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{APP_DIR};{os.environ.get('PATH', '')}"

import customtkinter as ctk
from dotenv import load_dotenv
from tkinter import filedialog, messagebox

load_dotenv(APP_DIR / ".env")

from transcriber import (
    AVAILABLE_MODELS,
    detect_optimal_device,
    load_whisper_model,
    transcribe_media_chunked,
    transcribe_media_gemini,
)
from utils import (
    check_ffmpeg_installed,
    export_srt,
    export_txt,
    format_duration,
    format_file_size,
    probe_media_file,
)

SUPPORTED_EXTENSIONS = [
    ("Semua File Media Audio/Video", "*.mp4;*.mkv;*.mp3;*.m4a;*.wav;*.flac;*.ogg;*.aac;*.webm;*.avi;*.mov;*.wmv;*.ts;*.m4v;*.wma;*.opus"),
    ("File Video (*.mp4, *.mkv, *.avi, *.webm, *.mov)", "*.mp4;*.mkv;*.avi;*.webm;*.mov;*.wmv;*.ts;*.m4v"),
    ("File Audio (*.mp3, *.m4a, *.wav, *.flac, *.ogg, *.aac)", "*.mp3;*.m4a;*.wav;*.flac;*.ogg;*.aac;*.opus;*.wma"),
    ("Semua File (*.*)", "*.*"),
]

POPULAR_LANGUAGES = [
    ("id", "Bahasa Indonesia (id)"),
    ("auto", "Deteksi Otomatis (auto)"),
    ("en", "English (en)"),
    ("jv", "Basa Jawa (jv)"),
    ("su", "Basa Sunda (su)"),
    ("ms", "Bahasa Melayu (ms)"),
    ("ja", "Japanese / 日本語 (ja)"),
    ("ko", "Korean / 한국어 (ko)"),
    ("zh", "Chinese / 中文 (zh)"),
    ("ar", "Arabic / العربية (ar)"),
    ("es", "Spanish (es)"),
    ("de", "German (de)"),
    ("fr", "French (fr)"),
]

MODEL_OPTIONS = [
    ("base", "base (~1 GB VRAM) - Standar & Cepat [Default]"),
    ("tiny", "tiny (~1 GB VRAM) - Super Ringan & Cepat"),
    ("small", "small (~2 GB VRAM) - Akurat (Rekomendasi GPU 4GB)"),
    ("medium", "medium (~5 GB VRAM) - Sangat Akurat (GPU >= 6GB)"),
    ("large-v3-turbo", "large-v3-turbo (~4 GB VRAM) - Rekomendasi Utama GPU >= 4GB"),
    ("turbo", "turbo (~4 GB VRAM) - Cepat & Akurasi Tinggi"),
    ("large-v3", "large-v3 (~10 GB VRAM) - Kualitas Maksimal"),
    ("gemini-4-hydrogen", "Google Gemini 4 Hydrogen Audio - Direkomendasikan"),
]


class TranskriptorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)

        self.title("Whisper Transkriptor Studio")
        self.geometry("1180x820")
        self.minsize(980, 700)

        self.is_running = False
        self.stop_requested = False
        self.transcribe_thread: Optional[threading.Thread] = None
        self.msg_queue = queue.Queue()

        self.current_media_path: Optional[Path] = None
        self.current_media_info: Dict[str, Any] = {}
        self.last_txt_path: Optional[Path] = None
        self.last_srt_path: Optional[Path] = None
        self.last_output_dir: Optional[Path] = None

        self.segment_rows: List[Dict[str, Any]] = []
        self.transcript_font_size = 14
        self.show_timestamps = True
        self.started_at: Optional[float] = None

        self._init_theme_colors()
        self._build_ui()

        self.after(200, self._check_system_hardware)
        self.after(100, self._process_queue)
        self.after(500, self._update_elapsed_timer)

    def _init_theme_colors(self):
        self.colors = {
            "bg": "#0B0E14",
            "card": "#131720",
            "surface": "#1A1F2C",
            "surface_hover": "#222736",
            "border": "#222736",
            "border_strong": "#2A3142",
            "fg": "#F8FAFC",
            "muted": "#94A3B8",
            "muted_2": "#64748B",
            "accent": "#6366F1",
            "accent_hover": "#4F46E5",
            "success": "#34D399",
            "success_bg": "#132A24",
            "danger": "#F87171",
            "danger_bg": "#2A1519",
            "warning": "#FBBF24",
        }
        self.configure(fg_color=self.colors["bg"])

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=360)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()

        sidebar = ctk.CTkFrame(
            self, fg_color=self.colors["bg"], corner_radius=0, border_width=0
        )
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(14, 7), pady=(8, 14))
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=0)
        sidebar.grid_rowconfigure(1, weight=0)
        sidebar.grid_rowconfigure(2, weight=1)

        self._build_media_card(sidebar)
        self._build_settings_card(sidebar)
        self._build_control_card(sidebar)

        workspace = ctk.CTkFrame(
            self,
            fg_color=self.colors["card"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
        )
        workspace.grid(row=1, column=1, sticky="nsew", padx=(7, 14), pady=(8, 14))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)
        workspace.grid_rowconfigure(2, weight=0)

        self._build_output_card(workspace)

    def _build_header(self):
        header = ctk.CTkFrame(
            self,
            fg_color=self.colors["card"],
            height=72,
            corner_radius=0,
            border_width=0,
        )
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=18, pady=12)

        ctk.CTkLabel(
            title_box,
            text="Whisper Transkriptor Studio",
            font=("Segoe UI", 18, "bold"),
            text_color=self.colors["fg"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="faster-whisper  •  VAD Active",
            font=("Segoe UI", 10),
            text_color=self.colors["muted"],
        ).pack(anchor="w", pady=(3, 0))

        self.lbl_hw_badge = ctk.CTkLabel(
            header,
            text="Mendeteksi perangkat...",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.colors["surface"],
            text_color=self.colors["muted"],
            corner_radius=8,
            padx=12,
            pady=7,
        )
        self.lbl_hw_badge.grid(row=0, column=1, sticky="e", padx=18)

    def _build_media_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=self.colors["card"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
        )
        card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="SUMBER MEDIA",
            font=("Segoe UI", 11, "bold"),
            text_color=self.colors["muted"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        file_row = ctk.CTkFrame(card, fg_color="transparent")
        file_row.grid(row=1, column=0, sticky="ew", padx=14)
        file_row.grid_columnconfigure(0, weight=1)

        self.ent_file = ctk.CTkEntry(
            file_row,
            placeholder_text="Pilih file audio atau video...",
            fg_color=self.colors["surface"],
            border_color=self.colors["border"],
            text_color=self.colors["fg"],
            height=38,
        )
        self.ent_file.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            file_row,
            text="Pilih File",
            width=88,
            height=38,
            fg_color=self.colors["surface_hover"],
            hover_color=self.colors["border_strong"],
            text_color=self.colors["fg"],
            corner_radius=8,
            command=self._on_browse_file,
        ).grid(row=0, column=1)

        self.lbl_media_meta = ctk.CTkLabel(
            card,
            text="Belum ada media yang dipilih",
            font=("Segoe UI", 11),
            text_color=self.colors["muted"],
            fg_color=self.colors["surface"],
            corner_radius=8,
            justify="left",
            anchor="w",
            padx=12,
            pady=10,
        )
        self.lbl_media_meta.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 8))

        out_row = ctk.CTkFrame(card, fg_color="transparent")
        out_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        out_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            out_row,
            text="Folder hasil",
            font=("Segoe UI", 10, "bold"),
            text_color=self.colors["muted"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.ent_output_dir = ctk.CTkEntry(
            out_row,
            placeholder_text="Folder output otomatis...",
            fg_color=self.colors["surface"],
            border_color=self.colors["border"],
            text_color=self.colors["fg"],
            height=36,
        )
        self.ent_output_dir.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            out_row,
            text="...",
            width=42,
            height=36,
            fg_color=self.colors["surface_hover"],
            hover_color=self.colors["border_strong"],
            text_color=self.colors["fg"],
            corner_radius=8,
            command=self._on_browse_output_dir,
        ).grid(row=1, column=1)

    def _build_settings_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=self.colors["card"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
        )
        card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            card,
            text="ENGINE",
            font=("Segoe UI", 11, "bold"),
            text_color=self.colors["muted"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))

        model_box = ctk.CTkFrame(card, fg_color="transparent")
        model_box.grid(row=1, column=0, sticky="ew", padx=(14, 6))
        ctk.CTkLabel(model_box, text="Model Whisper", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w")
        self.model_labels = {key: label for key, label in MODEL_OPTIONS}
        self.model_keys_by_label = {label: key for key, label in MODEL_OPTIONS}
        self.var_model = ctk.StringVar(value=self.model_labels["base"])
        self.cbo_model = ctk.CTkOptionMenu(
            model_box,
            variable=self.var_model,
            values=[opt[1] for opt in MODEL_OPTIONS],
            fg_color=self.colors["surface"],
            button_color=self.colors["surface_hover"],
            button_hover_color=self.colors["border_strong"],
            dropdown_fg_color=self.colors["surface"],
            text_color=self.colors["fg"],
            height=36,
            corner_radius=8,
        )
        self.cbo_model.pack(fill="x", pady=(5, 0))

        lang_box = ctk.CTkFrame(card, fg_color="transparent")
        lang_box.grid(row=1, column=1, sticky="ew", padx=(6, 14))
        ctk.CTkLabel(lang_box, text="Bahasa Audio", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w")
        self.var_lang = ctk.StringVar(value="Bahasa Indonesia (id)")
        self.cbo_lang = ctk.CTkOptionMenu(
            lang_box,
            variable=self.var_lang,
            values=[opt[1] for opt in POPULAR_LANGUAGES],
            fg_color=self.colors["surface"],
            button_color=self.colors["surface_hover"],
            button_hover_color=self.colors["border_strong"],
            dropdown_fg_color=self.colors["surface"],
            text_color=self.colors["fg"],
            height=36,
            corner_radius=8,
        )
        self.cbo_lang.pack(fill="x", pady=(5, 0))

        device_box = ctk.CTkFrame(card, fg_color="transparent")
        device_box.grid(row=2, column=0, sticky="ew", padx=(14, 6), pady=(8, 0))
        ctk.CTkLabel(device_box, text="Perangkat", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w")
        self.var_device = ctk.StringVar(value="auto")
        self.cbo_device = ctk.CTkOptionMenu(
            device_box, variable=self.var_device, values=["auto", "cuda", "cpu"],
            fg_color=self.colors["surface"], button_color=self.colors["surface_hover"],
            button_hover_color=self.colors["border_strong"], dropdown_fg_color=self.colors["surface"],
            text_color=self.colors["fg"], height=36, corner_radius=8,
        )
        self.cbo_device.pack(fill="x", pady=(5, 0))

        task_box = ctk.CTkFrame(card, fg_color="transparent")
        task_box.grid(row=2, column=1, sticky="ew", padx=(6, 14), pady=(8, 0))
        ctk.CTkLabel(task_box, text="Mode Tugas", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w")
        self.var_task = ctk.StringVar(value="transcribe")
        self.cbo_task = ctk.CTkOptionMenu(
            task_box, variable=self.var_task, values=["transcribe", "translate"],
            fg_color=self.colors["surface"], button_color=self.colors["surface_hover"],
            button_hover_color=self.colors["border_strong"], dropdown_fg_color=self.colors["surface"],
            text_color=self.colors["fg"], height=36, corner_radius=8,
        )
        self.cbo_task.pack(fill="x", pady=(5, 0))

        advanced = ctk.CTkFrame(card, fg_color="transparent")
        advanced.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 0))
        advanced.grid_columnconfigure(0, weight=0)
        advanced.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(advanced, text="Chunk (menit)", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).grid(row=0, column=0, sticky="w")
        self.var_chunk_size = ctk.StringVar(value="10")
        self.cbo_chunk = ctk.CTkOptionMenu(
            advanced, variable=self.var_chunk_size, values=["5", "10", "15", "20", "30"],
            width=78, height=32, fg_color=self.colors["surface"],
            button_color=self.colors["surface_hover"], button_hover_color=self.colors["border_strong"],
            dropdown_fg_color=self.colors["surface"], text_color=self.colors["fg"], corner_radius=8,
        )
        self.cbo_chunk.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.var_fp32 = ctk.BooleanVar(value=False)
        self.chk_fp32 = ctk.CTkSwitch(
            advanced,
            text="Paksa FP32",
            variable=self.var_fp32,
            fg_color=self.colors["surface_hover"],
            progress_color=self.colors["accent"],
            button_color=self.colors["muted"],
            button_hover_color=self.colors["fg"],
            text_color=self.colors["fg"],
            font=("Segoe UI", 10, "bold"),
        )
        self.chk_fp32.grid(row=1, column=1, sticky="e")

        prompt_box = ctk.CTkFrame(card, fg_color="transparent")
        prompt_box.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 12))
        ctk.CTkLabel(prompt_box, text="Petunjuk Kosakata", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w")
        self.ent_prompt = ctk.CTkEntry(
            prompt_box,
            placeholder_text="Contoh: istilah teknis, nama brand, bahasa daerah...",
            fg_color=self.colors["surface"],
            border_color=self.colors["border"],
            text_color=self.colors["fg"],
            height=36,
        )
        self.ent_prompt.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(prompt_box, text="API Key Gemini (opsional)", font=("Segoe UI", 10, "bold"), text_color=self.colors["muted"]).pack(anchor="w", pady=(10, 0))
        self.ent_gemini_key = ctk.CTkEntry(
            prompt_box,
            placeholder_text="Kosongkan jika memakai GEMINI_API_KEY",
            show="*",
            fg_color=self.colors["surface"],
            border_color=self.colors["border"],
            text_color=self.colors["fg"],
        )
        self.ent_gemini_key.pack(fill="x", pady=(5, 0))

    def _build_control_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=self.colors["card"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
        )
        card.grid(row=2, column=0, sticky="sew")
        card.grid_columnconfigure(0, weight=1)

        action_row = ctk.CTkFrame(card, fg_color="transparent")
        action_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        action_row.grid_columnconfigure(0, weight=1)

        self.btn_start = ctk.CTkButton(
            action_row,
            text="▶  MULAI TRANSKRIPSI",
            height=42,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            text_color="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            corner_radius=8,
            command=self._on_start_transcription,
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 7))

        self.btn_stop = ctk.CTkButton(
            action_row,
            text="Stop",
            width=72,
            height=42,
            state="disabled",
            fg_color=self.colors["danger_bg"],
            hover_color="#3A1B20",
            text_color=self.colors["danger"],
            border_width=1,
            border_color=self.colors["danger"],
            font=("Segoe UI", 10, "bold"),
            corner_radius=8,
            command=self._on_stop_transcription,
        )
        self.btn_stop.grid(row=0, column=1)

        status_row = ctk.CTkFrame(card, fg_color="transparent")
        status_row.grid(row=1, column=0, sticky="ew", padx=12)
        status_row.grid_columnconfigure(0, weight=1)

        self.lbl_status = ctk.CTkLabel(
            status_row,
            text="Siap untuk memulai",
            font=("Segoe UI", 10),
            text_color=self.colors["muted"],
            anchor="w",
        )
        self.lbl_status.grid(row=0, column=0, sticky="w")

        self.lbl_elapsed = ctk.CTkLabel(
            status_row,
            text="0:00",
            font=("Consolas", 10),
            text_color=self.colors["muted_2"],
        )
        self.lbl_elapsed.grid(row=0, column=1, sticky="e")

        self.progress_bar = ctk.CTkProgressBar(
            card,
            height=6,
            fg_color=self.colors["surface"],
            progress_color=self.colors["accent"],
            corner_radius=6,
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        self.progress_bar.set(0)

    def _build_output_card(self, parent):
        toolbar = ctk.CTkFrame(
            parent,
            fg_color=self.colors["card"],
            height=52,
            corner_radius=10,
        )
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        toolbar.grid_columnconfigure(1, weight=1)

        self.view_switch = ctk.CTkSegmentedButton(
            toolbar,
            values=["Transkrip", "Log Sistem"],
            selected_color=self.colors["accent"],
            selected_hover_color=self.colors["accent_hover"],
            unselected_color=self.colors["surface"],
            unselected_hover_color=self.colors["surface_hover"],
            text_color=self.colors["muted"],
            text_color_disabled=self.colors["muted_2"],
            fg_color=self.colors["surface"],
            height=34,
            corner_radius=8,
            command=self._on_view_switch,
        )
        self.view_switch.set("Transkrip")
        self.view_switch.grid(row=0, column=0, padx=(8, 10), pady=8)

        control_row = ctk.CTkFrame(toolbar, fg_color="transparent")
        control_row.grid(row=0, column=2, sticky="e", padx=8)

        ctk.CTkButton(
            control_row, text="A-", width=42, height=34,
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            text_color=self.colors["fg"], corner_radius=8,
            command=self._decrease_font,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            control_row, text="A+", width=42, height=34,
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            text_color=self.colors["fg"], corner_radius=8,
            command=self._increase_font,
        ).pack(side="left", padx=4)

        self.timestamp_switch = ctk.CTkSwitch(
            control_row,
            text="Timestamp",
            variable=ctk.BooleanVar(value=True),
            fg_color=self.colors["surface_hover"],
            progress_color=self.colors["accent"],
            button_color=self.colors["muted"],
            text_color=self.colors["muted"],
            font=("Segoe UI", 10),
            command=self._on_timestamp_toggle,
        )
        self.timestamp_switch.select()
        self.timestamp_switch.pack(side="left", padx=8)

        self.lbl_counter = ctk.CTkLabel(
            control_row,
            text="0 kata • 0 karakter",
            font=("Consolas", 10),
            fg_color=self.colors["surface"],
            text_color=self.colors["muted"],
            corner_radius=8,
            padx=10,
            pady=7,
        )
        self.lbl_counter.pack(side="left", padx=(4, 0))

        canvas = ctk.CTkFrame(
            parent,
            fg_color=self.colors["surface"],
            corner_radius=10,
            border_width=1,
            border_color=self.colors["border"],
        )
        canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        canvas.grid_columnconfigure(0, weight=1)
        canvas.grid_rowconfigure(0, weight=1)

        self.txt_transcript = ctk.CTkTextbox(
            canvas,
            wrap="word",
            fg_color=self.colors["surface"],
            text_color=self.colors["fg"],
            font=("Segoe UI", self.transcript_font_size),
            border_width=0,
            corner_radius=10,
            padx=20,
            pady=20,
        )
        self.txt_transcript.grid(row=0, column=0, sticky="nsew")

        self.txt_logs = ctk.CTkTextbox(
            canvas,
            wrap="word",
            fg_color=self.colors["surface"],
            text_color=self.colors["muted"],
            font=("Consolas", 12),
            border_width=0,
            corner_radius=10,
            padx=20,
            pady=20,
        )
        self.txt_logs.grid(row=0, column=0, sticky="nsew")

        self._build_empty_state(canvas)

        self.empty_state.lift()
        self.txt_logs.lower()
        self.txt_transcript.lower()

        bottom = ctk.CTkFrame(parent, fg_color="transparent", height=50)
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.grid_columnconfigure(1, weight=1)

        left_actions = ctk.CTkFrame(bottom, fg_color="transparent")
        left_actions.grid(row=0, column=0, sticky="w")

        self.btn_open_txt = ctk.CTkButton(
            left_actions, text="📄  .TXT", width=88, height=36, state="disabled",
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            border_width=1, border_color=self.colors["border_strong"],
            text_color=self.colors["muted"], corner_radius=8, command=self._on_open_txt,
        )
        self.btn_open_txt.pack(side="left", padx=(0, 6))

        self.btn_open_srt = ctk.CTkButton(
            left_actions, text="🎬  .SRT", width=88, height=36, state="disabled",
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            border_width=1, border_color=self.colors["border_strong"],
            text_color=self.colors["muted"], corner_radius=8, command=self._on_open_srt,
        )
        self.btn_open_srt.pack(side="left", padx=6)

        self.btn_open_dir = ctk.CTkButton(
            left_actions, text="📁  Folder", width=92, height=36, state="disabled",
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            border_width=1, border_color=self.colors["border_strong"],
            text_color=self.colors["muted"], corner_radius=8, command=self._on_open_folder,
        )
        self.btn_open_dir.pack(side="left", padx=6)

        right_actions = ctk.CTkFrame(bottom, fg_color="transparent")
        right_actions.grid(row=0, column=2, sticky="e")

        self.btn_copy = ctk.CTkButton(
            right_actions, text="📋  Salin", width=88, height=36,
            fg_color="transparent", hover_color=self.colors["surface_hover"],
            border_width=1, border_color=self.colors["accent"],
            text_color=self.colors["fg"], corner_radius=8, command=self._on_copy_transcript,
        )
        self.btn_copy.pack(side="left", padx=(0, 6))

        self.btn_clear = ctk.CTkButton(
            right_actions, text="🗑️  Bersihkan", width=112, height=36,
            fg_color=self.colors["surface"], hover_color=self.colors["surface_hover"],
            text_color=self.colors["muted"], corner_radius=8, command=self._on_clear_output,
        )
        self.btn_clear.pack(side="left", padx=6)

    def _build_empty_state(self, parent):
        self.empty_state = ctk.CTkFrame(
            parent, fg_color=self.colors["surface"], corner_radius=12
        )
        self.empty_state.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self.empty_state,
            text="",
            width=56,
            height=56,
            fg_color=self.colors["surface_hover"],
            text_color=self.colors["accent"],
            corner_radius=28,
            text_color_disabled=self.colors["accent"],
        ).grid(row=0, column=0, pady=(22, 12))
        ctk.CTkLabel(
            self.empty_state,
            text="Siap Mentranskrip Audio & Video",
            font=("Segoe UI", 16, "bold"),
            text_color=self.colors["fg"],
        ).grid(row=1, column=0, padx=38)

        ctk.CTkLabel(
            self.empty_state,
            text="Pilih media untuk memulai sesi transkripsi",
            font=("Segoe UI", 11),
            text_color=self.colors["muted"],
        ).grid(row=2, column=0, pady=(6, 12))

        format_row = ctk.CTkFrame(self.empty_state, fg_color="transparent")
        format_row.grid(row=3, column=0, pady=(0, 14))
        for label in ("MP4", "MKV", "MP3", "WAV", "M4A"):
            ctk.CTkLabel(
                format_row,
                text=label,
                font=("Segoe UI", 9, "bold"),
                fg_color=self.colors["surface_hover"],
                text_color=self.colors["muted"],
                corner_radius=7,
                padx=8,
                pady=5,
            ).pack(side="left", padx=3)

        ctk.CTkButton(
            self.empty_state,
            text="Pilih File",
            height=38,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            text_color="#FFFFFF",
            corner_radius=8,
            command=self._on_browse_file,
        ).grid(row=4, column=0, pady=(0, 20))

    # EVENT HANDLERS & LOGIC
    def _check_system_hardware(self):
        self.lbl_hw_badge.configure(
            text="Perangkat akan dideteksi saat mulai",
            text_color=self.colors["muted"],
        )

        def worker():
            try:
                ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
                self.msg_queue.put(("STARTUP_CHECKED", (ffmpeg_ok, ffmpeg_msg)))
            except Exception as exc:
                self.msg_queue.put(("STARTUP_CHECK_ERR", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_browse_file(self):
        selected_file = filedialog.askopenfilename(
            title="Pilih File Audio atau Video",
            filetypes=SUPPORTED_EXTENSIONS
        )
        if not selected_file:
            return

        file_path = Path(selected_file).resolve()
        self.current_media_path = file_path
        self.ent_file.delete(0, "end")
        self.ent_file.insert(0, str(file_path))

        if not self.ent_output_dir.get().strip():
            self.ent_output_dir.delete(0, "end")
            self.ent_output_dir.insert(0, str(file_path.parent))

        self._inspect_file_async(file_path)

    def _inspect_file_async(self, file_path: Path):
        self.lbl_media_meta.configure(
            text="Memeriksa metadata file media...",
            text_color=self.colors["muted"],
        )

        def worker():
            try:
                info = probe_media_file(file_path)
                self.msg_queue.put(("MEDIA_PROBED", info))
            except Exception as e:
                self.msg_queue.put(("MEDIA_PROBED_ERR", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_browse_output_dir(self):
        selected_dir = filedialog.askdirectory(title="Pilih Direktori Penyimpanan Output")
        if selected_dir:
            self.ent_output_dir.delete(0, "end")
            self.ent_output_dir.insert(0, str(Path(selected_dir).resolve()))

    def _on_start_transcription(self):
        file_str = self.ent_file.get().strip()
        if not file_str:
            messagebox.showerror("Error", "Silakan pilih file media audio/video terlebih dahulu!")
            return

        file_path = Path(file_str).resolve()
        if not file_path.is_file():
            messagebox.showerror("Error", f"File tidak ditemukan:\n{file_path}")
            return

        output_dir_str = self.ent_output_dir.get().strip()
        out_dir = Path(output_dir_str).resolve() if output_dir_str else file_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = out_dir

        stem = file_path.stem
        self.last_txt_path = out_dir / f"{stem}.txt"
        self.last_srt_path = out_dir / f"{stem}.srt"

        raw_lang = self.var_lang.get()
        if "(" in raw_lang and ")" in raw_lang:
            lang_code = raw_lang.split("(")[-1].replace(")", "").strip()
        else:
            lang_code = raw_lang.strip()

        params = {
            "file_path": file_path,
            "output_dir": out_dir,
            "txt_path": self.last_txt_path,
            "srt_path": self.last_srt_path,
            "model_name": self.model_keys_by_label.get(self.var_model.get().strip(), self.var_model.get().strip()),
            "gemini_api_key": self.ent_gemini_key.get().strip() or None,
            "speaker_a_name": os.getenv("SPEAKER_A_NAME", "Si A").strip() or "Si A",
            "speaker_b_name": os.getenv("SPEAKER_B_NAME", "Si B").strip() or "Si B",
            "language": lang_code,
            "device": self.var_device.get().strip(),
            "task": self.var_task.get().strip(),
            "chunk_size_min": int(self.var_chunk_size.get().strip()),
            "force_fp32": self.var_fp32.get(),
            "initial_prompt": self.ent_prompt.get().strip() or None,
            "duration": self.current_media_info.get("duration", 0.0),
            "duration_str": self.current_media_info.get("duration_str", "Tidak diketahui"),
        }

        self.is_running = True
        self.stop_requested = False
        self.started_at = time.time()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(
            state="normal",
            fg_color=self.colors["danger_bg"],
            text_color=self.colors["danger"],
        )
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Memulai sesi transkripsi...", text_color=self.colors["accent"])
        self.empty_state.place_forget()
        self.txt_transcript.lift()

        self._log_system(f"Memulai sesi transkripsi untuk '{file_path.name}'...")
        self._log_system(f"Model: {params['model_name']} | Bahasa: {params['language']} | Device: {params['device']}")

        self.transcribe_thread = threading.Thread(target=self._run_transcription, args=(params,), daemon=True)
        self.transcribe_thread.start()

    def _on_stop_transcription(self):
        if self.is_running:
            self.stop_requested = True
            self.lbl_status.configure(
                text="Menghentikan proses (menyimpan chunk aktif)...",
                text_color=self.colors["warning"],
            )
            self._log_system("Permintaan penghentian diterima. Menyimpan progress terakhir...")
            self.btn_stop.configure(state="disabled")

    def _run_transcription(self, params: Dict[str, Any]):
        file_path: Path = params["file_path"]
        txt_path: Path = params["txt_path"]
        srt_path: Path = params["srt_path"]
        model_name: str = params["model_name"]
        device_req: str = params["device"]
        chunk_min: int = params["chunk_size_min"]

        try:
            if model_name == "gemini-4-hydrogen":
                self.msg_queue.put(("STATUS", "Mengunggah media ke Google Gemini..."))
                result = transcribe_media_gemini(
                    file_path=file_path,
                    language=params["language"],
                    task=params["task"],
                    initial_prompt=params["initial_prompt"],
                    api_key=params["gemini_api_key"],
                    total_duration=params["duration"],
                    speaker_a_name=params["speaker_a_name"],
                    speaker_b_name=params["speaker_b_name"],
                )
                all_segs = result["segments"]
                full_txt = result["text"]
                export_txt(full_txt, all_segs, txt_path, True, file_path.name, params["duration_str"])
                export_srt(all_segs, srt_path)
                for seg in all_segs:
                    self.msg_queue.put(("SEGMENT", seg))
                self.msg_queue.put(("CHUNK_DONE", (1, 1, 100, txt_path, srt_path)))
                self.msg_queue.put(("FINISHED_SUCCESS", (txt_path, srt_path, result.get("language", "auto"))))
                return

            device, fp16, desc = detect_optimal_device(
                requested_device=device_req,
                force_fp32=params["force_fp32"]
            )
            self.msg_queue.put(("LOG", f"Perangkat terpilih: {desc}"))

            self.msg_queue.put(("STATUS", f"Memuat model Whisper '{model_name}' ke {device.upper()}..."))
            start_load = time.time()
            model = load_whisper_model(model_name=model_name, device=device, force_fp32=params["force_fp32"])
            load_elapsed = time.time() - start_load
            self.msg_queue.put(("LOG", f"Model Whisper '{model_name}' siap dalam {load_elapsed:.2f} detik."))

            self.msg_queue.put(("STATUS", "Memulai transkripsi media..."))

            def stop_check() -> bool:
                return self.stop_requested

            def on_chunk_start(idx: int, total: int, time_range: str):
                pct = int(((idx - 1) / total) * 100)
                self.msg_queue.put(("CHUNK_START", (idx, total, time_range, pct)))

            def on_segment(seg: Dict[str, Any]):
                self.msg_queue.put(("SEGMENT", seg))

            def on_chunk_completed(idx: int, total: int, all_segs: List[Dict[str, Any]], full_txt: str):
                export_txt(
                    text=full_txt,
                    segments=all_segs,
                    output_path=txt_path,
                    include_timestamps=True,
                    file_name=file_path.name,
                    duration_str=params["duration_str"],
                )
                export_srt(segments=all_segs, output_path=srt_path)
                pct = int((idx / total) * 100)
                self.msg_queue.put(("CHUNK_DONE", (idx, total, pct, txt_path, srt_path)))

            result = transcribe_media_chunked(
                model=model,
                file_path=file_path,
                total_duration=params["duration"] or 0.0,
                chunk_size_sec=float(chunk_min * 60),
                language=params["language"],
                task=params["task"],
                verbose=False,
                fp16=fp16,
                initial_prompt=params["initial_prompt"],
                on_chunk_completed=on_chunk_completed,
                on_segment=on_segment,
                on_chunk_start=on_chunk_start,
                stop_check=stop_check,
            )

            if self.stop_requested:
                self.msg_queue.put(("FINISHED_STOPPED", (txt_path, srt_path)))
            else:
                self.msg_queue.put(("FINISHED_SUCCESS", (txt_path, srt_path, result.get("language", "id"))))

        except Exception as e:
            self.msg_queue.put(("ERROR", str(e)))

    def _process_queue(self):
        try:
            while True:
                event, data = self.msg_queue.get_nowait()

                if event == "MEDIA_PROBED":
                    self.current_media_info = data
                    dur_str = data.get("duration_str", "-")
                    size_str = data.get("file_size_str", "-")
                    codec = data.get("audio_codec") or "Tidak diketahui"
                    channels = data.get("channels") or "-"
                    rate = data.get("sample_rate") or "-"
                    name = self.current_media_path.name if self.current_media_path else "Media"
                    self.lbl_media_meta.configure(
                        text=f"{name}\n{size_str}  •  {dur_str}\n{codec} • {rate} Hz • {channels} ch",
                        text_color=self.colors["fg"],
                    )

                elif event == "MEDIA_PROBED_ERR":
                    self.lbl_media_meta.configure(
                        text=f"Gagal membaca metadata:\n{data}",
                        text_color=self.colors["danger"],
                    )

                elif event == "STARTUP_CHECKED":
                    ffmpeg_ok, ffmpeg_msg = data
                    if not ffmpeg_ok:
                        self._log_system(f"PERINGATAN: FFmpeg tidak terdeteksi!\n{ffmpeg_msg}")
                        messagebox.showwarning("FFmpeg Diperlukan", ffmpeg_msg)
                    else:
                        self._log_system("FFmpeg & FFprobe terdeteksi dan siap digunakan.")
                    self.lbl_hw_badge.configure(
                        text="Perangkat siap; GPU dideteksi saat mulai",
                        fg_color=self.colors["surface"],
                        text_color=self.colors["muted"],
                    )

                elif event == "STARTUP_CHECK_ERR":
                    self.lbl_hw_badge.configure(
                        text="Pemeriksaan startup gagal",
                        fg_color=self.colors["surface"],
                        text_color=self.colors["warning"],
                    )
                    self._log_system(f"Gagal mendeteksi perangkat: {data}")

                elif event == "LOG":
                    self._log_system(data)

                elif event == "STATUS":
                    self.lbl_status.configure(text=data, text_color=self.colors["accent"])

                elif event == "CHUNK_START":
                    idx, total, time_range, pct = data
                    self.progress_bar.set(pct / 100)
                    self.lbl_status.configure(
                        text=f"Mentranskrip segmen {idx}/{total}  •  {time_range}",
                        text_color=self.colors["accent"],
                    )
                    self._log_system(f"Memproses Bagian [{idx}/{total}]: {time_range}")

                elif event == "SEGMENT":
                    seg = data
                    seg_text = seg.get("text", "").strip()
                    if seg_text:
                        self.segment_rows.append(seg)
                        self._render_transcript()
                        self.empty_state.place_forget()
                        self.txt_transcript.lift()

                elif event == "CHUNK_DONE":
                    idx, total, pct, txt_p, srt_p = data
                    self.progress_bar.set(pct / 100)
                    self._log_system(f"Autosave Bagian {idx}/{total} tersimpan ke '{txt_p.name}'")
                    self.btn_open_txt.configure(state="normal")
                    self.btn_open_srt.configure(state="normal")
                    self.btn_open_dir.configure(state="normal")

                elif event == "FINISHED_SUCCESS":
                    txt_p, srt_p, detected_lang = data
                    self._on_transcription_finished(
                        f"Selesai • Bahasa: {detected_lang} • Hasil tersimpan",
                        success=True,
                    )
                    messagebox.showinfo(
                        "Transkripsi Berhasil!",
                        f"Transkripsi selesai dengan sukses!\n\n"
                        f"File Hasil:\n1. {txt_p.name}\n2. {srt_p.name}\n\n"
                        f"Disimpan di folder:\n{txt_p.parent}",
                    )

                elif event == "FINISHED_STOPPED":
                    txt_p, srt_p = data
                    self._on_transcription_finished(
                        "Proses dihentikan • Hasil parsial tersimpan",
                        success=False,
                    )

                elif event == "ERROR":
                    self._on_transcription_finished(f"Terjadi kesalahan: {data}", success=False)
                    messagebox.showerror("Terjadi Kesalahan", f"Transkripsi gagal:\n\n{data}")

                self.msg_queue.task_done()
        except queue.Empty:
            pass

        self.after(80, self._process_queue)

    def _on_transcription_finished(self, status_msg: str, success: bool):
        self.is_running = False
        self.started_at = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        color = self.colors["success"] if success else self.colors["warning"]
        self.lbl_status.configure(text=status_msg, text_color=color)
        self.progress_bar.set(1 if success else self.progress_bar.get())
        self._log_system(status_msg)

    def _log_system(self, message: str):
        now_str = time.strftime("%H:%M:%S")
        self.txt_logs.insert("end", f"[{now_str}] {message}\n")
        self.txt_logs.see("end")

    def _render_transcript(self):
        self.txt_transcript.configure(font=("Segoe UI", self.transcript_font_size))
        self.txt_transcript.delete("1.0", "end")
        parts: List[str] = []
        for seg in self.segment_rows:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            word_count = len(seg.get("words", [])) or len(seg_text.split())
            if self.show_timestamps:
                t_start = format_duration(seg.get("start", 0.0))
                t_end = format_duration(seg.get("end", 0.0))
                parts.append(f"[{t_start} --> {t_end}] ({word_count}w) {seg_text}\n")
            else:
                parts.append(f"{seg_text}\n")
        self.txt_transcript.insert("1.0", "".join(parts))
        self.txt_transcript.see("end")
        self._update_counter()

    def _update_counter(self):
        text = self.txt_transcript.get("1.0", "end").strip()
        words = len(text.split()) if text else 0
        chars = len(text)
        self.lbl_counter.configure(text=f"{words} kata • {chars} karakter")

    def _on_view_switch(self, value: str):
        if value == "Log Sistem":
            self.txt_logs.lift()
            self.empty_state.place_forget()
        else:
            self.txt_transcript.lift()
            if not self.segment_rows:
                self.empty_state.place(relx=0.5, rely=0.5, anchor="center")
                self.empty_state.lift()

    def _decrease_font(self):
        self.transcript_font_size = max(11, self.transcript_font_size - 1)
        self._render_transcript()

    def _increase_font(self):
        self.transcript_font_size = min(22, self.transcript_font_size + 1)
        self._render_transcript()

    def _on_timestamp_toggle(self):
        self.show_timestamps = bool(self.timestamp_switch.get())
        self._render_transcript()

    def _update_elapsed_timer(self):
        if self.is_running and self.started_at:
            elapsed = int(time.time() - self.started_at)
            minutes, seconds = divmod(elapsed, 60)
            self.lbl_elapsed.configure(text=f"{minutes}:{seconds:02d}")
        self.after(500, self._update_elapsed_timer)

    def _on_open_txt(self):
        if self.last_txt_path and self.last_txt_path.is_file():
            self._open_file_native(self.last_txt_path)
        else:
            messagebox.showinfo("Informasi", "File .TXT belum tersedia.")

    def _on_open_srt(self):
        if self.last_srt_path and self.last_srt_path.is_file():
            self._open_file_native(self.last_srt_path)
        else:
            messagebox.showinfo("Informasi", "File .SRT belum tersedia.")

    def _on_open_folder(self):
        target_dir = self.last_output_dir or (self.current_media_path.parent if self.current_media_path else None)
        if target_dir and target_dir.is_dir():
            if sys.platform == "win32":
                os.startfile(str(target_dir))
            else:
                subprocess.Popen(["xdg-open", str(target_dir)])
        else:
            messagebox.showinfo("Informasi", "Folder belum tersedia.")

    def _open_file_native(self, file_path: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(file_path))
            else:
                subprocess.Popen(["xdg-open", str(file_path)])
        except Exception as e:
            messagebox.showerror("Gagal Membuka File", f"Tidak dapat membuka file:\n{e}")

    def _on_copy_transcript(self):
        text = self.txt_transcript.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Info", "Transkrip masih kosong.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Tersalin", "Seluruh teks transkrip berhasil disalin ke Clipboard!")

    def _on_clear_output(self):
        self.segment_rows.clear()
        self.txt_transcript.delete("1.0", "end")
        self.txt_logs.delete("1.0", "end")
        self.lbl_status.configure(text="Layar dibersihkan.", text_color=self.colors["muted"])
        self.lbl_elapsed.configure(text="0:00")
        self.progress_bar.set(0)
        self._update_counter()
        self.empty_state.place(relx=0.5, rely=0.5, anchor="center")
        self.empty_state.lift()


def main():
    if sys.platform == "win32":
        try:
            if sys.stdout and hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    app = TranskriptorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
