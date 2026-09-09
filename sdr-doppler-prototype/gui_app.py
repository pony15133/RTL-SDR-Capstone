"""Desktop control panel for the SDR Doppler Prototype.

A Tkinter front-end over the project's existing CLI scripts (train_model.py,
src/main.py) plus the repo-root auto-capture connector (pipeline.py) - so a
full capture -> detect -> database run, or a manual detection/training pass,
can be driven without touching a terminal. Each run's summary JSON and
spectrogram PNG(s) land in their own timestamped session folder (see
src/storage.py's ensure_session_output_dir), which the Results and Image
Preview tabs below browse.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, Toplevel, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (
    DB_PATH,
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_MAX_SMOOTHNESS_HZ,
    DEFAULT_ML_MODEL_PATH,
    DEFAULT_MIN_DRIFT_HZ,
    DEFAULT_MIN_VALID_RATIO,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SNR_THRESHOLD_DB,
)
from storage import format_detection_summary, format_training_summary

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PIPELINE_SCRIPT = PROJECT_ROOT / "pipeline.py"


class SDRDopplerGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("SDR Doppler Prototype")
        self.root.geometry("1150x820")
        self.root.minsize(980, 650)

        # Auto Capture (pipeline.py: recorder -> detect -> database)
        self.auto_satellite_var = StringVar(value="METEOR-M2-4")
        self.auto_frequency_var = StringVar(value=str(DEFAULT_CENTER_FREQ_HZ))
        self.auto_sample_rate_var = StringVar(value=str(DEFAULT_SAMPLE_RATE_HZ))
        self.auto_gain_var = StringVar(value="auto")
        self.auto_duration_var = StringVar(value="60")
        self.auto_norad_id_var = StringVar(value="")
        self.auto_recording_dir_var = StringVar(value=str(PROJECT_ROOT / "iq-recorder" / "recordings"))
        self.auto_db_path_var = StringVar(value=str(DB_PATH))
        self.auto_output_dir_var = StringVar(value=str(APP_DIR / "data" / "results"))
        self.auto_simulate_var = BooleanVar(value=True)
        self.auto_save_image_var = BooleanVar(value=False)
        self.auto_no_ml_var = BooleanVar(value=False)

        # Train Model / Run Detection / Open Results Folder
        self.dataset_var = StringVar(value=str(APP_DIR / "data" / "training" / "synthetic_example.csv"))
        self.model_var = StringVar(value=str(APP_DIR / "models" / "random_forest.joblib"))
        self.input_var = StringVar(value=str(APP_DIR / "data" / "raw" / "sample.npy"))
        self.output_dir_var = StringVar(value=str(APP_DIR / "data" / "results"))
        self.db_path_var = StringVar(value=str(DB_PATH))
        self.sample_rate_var = StringVar(value=str(DEFAULT_SAMPLE_RATE_HZ))
        self.center_freq_var = StringVar(value=str(DEFAULT_CENTER_FREQ_HZ))
        self.nperseg_var = StringVar(value=str(DEFAULT_NPERSEG))
        self.noverlap_var = StringVar(value=str(DEFAULT_NOVERLAP))
        self.binary_dtype_var = StringVar(value="complex64")
        self.snr_threshold_db_var = StringVar(value=str(DEFAULT_SNR_THRESHOLD_DB))
        self.min_valid_ratio_var = StringVar(value=str(DEFAULT_MIN_VALID_RATIO))
        self.min_drift_hz_var = StringVar(value=str(DEFAULT_MIN_DRIFT_HZ))
        self.max_smoothness_hz_var = StringVar(value=str(DEFAULT_MAX_SMOOTHNESS_HZ))
        self.chunk_size_var = StringVar(value="250000")
        self.chunk_overlap_var = StringVar(value="0")
        self.status_var = StringVar(value="Ready")
        self.progress_text_var = StringVar(value="Idle")
        self.allow_synthetic_var = BooleanVar(value=True)
        self.save_image_var = BooleanVar(value=False)
        self.no_ml_var = BooleanVar(value=False)
        self.image_series = []
        self.current_image_index = 0

        self._build_ui()
        self._refresh_result_panel()
        self._refresh_image_preview()

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        header = ttk.Label(self.root, text="SDR Doppler Prototype", font=("Segoe UI", 16, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.function_tabs = ttk.Notebook(self.root)
        self.function_tabs.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))

        self.func_panels = {}
        for name in ["Auto Capture", "Train Model", "Run Detection", "Open Results Folder"]:
            frame = ttk.Frame(self.function_tabs, padding=(16, 10, 16, 10))
            frame.grid_columnconfigure(1, weight=1)
            self.function_tabs.add(frame, text=name)
            self.func_panels[name] = frame
            self._populate_function_tab(frame, name)

        body = ttk.Notebook(self.root)
        body.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        results_tab = ttk.Frame(body)
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(0, weight=1)
        body.add(results_tab, text="Results")

        self.results_box = ScrolledText(results_tab, wrap="word", state="disabled", height=18)
        self.results_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        image_tab = ttk.Frame(body)
        image_tab.grid_columnconfigure(0, weight=1)
        image_tab.grid_rowconfigure(0, weight=1)
        body.add(image_tab, text="Image Preview")

        nav_bar = ttk.Frame(image_tab)
        nav_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))
        self.image_prev_button = ttk.Button(nav_bar, text="Previous", command=self._show_previous_image)
        self.image_prev_button.pack(side="left")
        self.image_counter_var = StringVar(value="0 / 0")
        ttk.Label(nav_bar, textvariable=self.image_counter_var, width=16, anchor="center").pack(side="left", padx=12)
        self.image_next_button = ttk.Button(nav_bar, text="Next", command=self._show_next_image)
        self.image_next_button.pack(side="left")

        self.image_label = ttk.Label(image_tab, text="No PNG preview available yet.", anchor="center")
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        status_bar = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.grid_columnconfigure(0, weight=1)
        ttk.Label(status_bar, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_bar, textvariable=self.status_var, foreground="#005a9c", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(status_bar, textvariable=self.progress_text_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.progress_bar = ttk.Progressbar(status_bar, orient="horizontal", length=280, mode="indeterminate")
        self.progress_bar.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

    def _populate_function_tab(self, parent, selected):
        if selected == "Auto Capture":
            self._add_field_row_to_parent(parent, "Satellite/target name", self.auto_satellite_var)
            self._add_field_row_to_parent(parent, "Frequency (Hz)", self.auto_frequency_var)
            self._add_field_row_to_parent(parent, "Sample rate (Hz)", self.auto_sample_rate_var)
            self._add_field_row_to_parent(parent, "Gain (dB or 'auto')", self.auto_gain_var)
            self._add_field_row_to_parent(parent, "Duration (s)", self.auto_duration_var)
            self._add_field_row_to_parent(parent, "NORAD ID (optional)", self.auto_norad_id_var)
            self._add_field_row_to_parent(parent, "Recording folder", self.auto_recording_dir_var, browse_func=self._choose_auto_recording_dir)
            self._add_field_row_to_parent(parent, "Database path", self.auto_db_path_var, browse_func=self._choose_auto_db_path)
            self._add_field_row_to_parent(parent, "Detection output folder", self.auto_output_dir_var, browse_func=self._choose_auto_output_dir)
            self._add_checkbox_row_to_parent(parent, "Simulate (no hardware required)", self.auto_simulate_var)
            self._add_checkbox_row_to_parent(parent, "Save spectrogram image", self.auto_save_image_var)
            self._add_checkbox_row_to_parent(parent, "Skip ML detection", self.auto_no_ml_var)
            self._add_info_label_to_parent(
                parent,
                "Runs pipeline.py: records one pass with the RTL-SDR recorder, then feeds it straight "
                "through detection into the database - the auto-capture connector. Leave Simulate checked "
                "to try it without hardware attached.",
            )
            ttk.Button(parent, text="Run Auto Capture", command=lambda: self._run_selected_function("Auto Capture")).pack(anchor="w", pady=(12, 0))
        elif selected == "Train Model":
            self._add_field_row_to_parent(parent, "Dataset CSV", self.dataset_var, browse_func=self._choose_dataset)
            self._add_field_row_to_parent(parent, "Model output", self.model_var, browse_func=self._choose_model_path)
            self._add_checkbox_row_to_parent(parent, "Allow synthetic data", self.allow_synthetic_var)
            self._add_info_label_to_parent(parent, "This function trains the Random Forest model from a labelled CSV file.")
            ttk.Button(parent, text="Run Training", command=lambda: self._run_selected_function("Train Model")).pack(anchor="w", pady=(12, 0))
        elif selected == "Run Detection":
            self._add_field_row_to_parent(parent, "Capture input", self.input_var, browse_func=self._choose_input)
            self._add_field_row_to_parent(parent, "Output folder", self.output_dir_var, browse_func=self._choose_output_dir)
            self._add_field_row_to_parent(parent, "Database path", self.db_path_var, browse_func=self._choose_db_path)
            self._add_field_row_to_parent(parent, "Model path", self.model_var, browse_func=self._choose_model_path)
            self._add_field_row_to_parent(parent, "Sample rate (Hz)", self.sample_rate_var)
            self._add_field_row_to_parent(parent, "Center freq (Hz)", self.center_freq_var)
            self._add_field_row_to_parent(parent, "nperseg", self.nperseg_var)
            self._add_field_row_to_parent(parent, "noverlap", self.noverlap_var)
            self._add_field_row_to_parent(parent, "Binary dtype", self.binary_dtype_var)
            self._add_field_row_to_parent(parent, "SNR threshold (dB)", self.snr_threshold_db_var)
            self._add_field_row_to_parent(parent, "Min valid ratio", self.min_valid_ratio_var)
            self._add_field_row_to_parent(parent, "Min drift (Hz)", self.min_drift_hz_var)
            self._add_field_row_to_parent(parent, "Max smoothness (Hz)", self.max_smoothness_hz_var)
            self._add_checkbox_row_to_parent(parent, "Save spectrogram image", self.save_image_var)
            self._add_field_row_to_parent(parent, "Image chunk size", self.chunk_size_var)
            self._add_field_row_to_parent(parent, "Image chunk overlap", self.chunk_overlap_var)
            self._add_checkbox_row_to_parent(parent, "Skip ML detection", self.no_ml_var)
            self._add_info_label_to_parent(parent, "These values match the CLI defaults in the parser and can be tuned for each run.")
            ttk.Button(parent, text="Run Detection", command=lambda: self._run_selected_function("Run Detection")).pack(anchor="w", pady=(12, 0))
        elif selected == "Open Results Folder":
            self._add_field_row_to_parent(parent, "Results folder", self.output_dir_var, browse_func=self._choose_output_dir)
            self._add_info_label_to_parent(parent, "Opens the output folder containing summary JSON, images, and model artifacts.")
            ttk.Button(parent, text="Open Results Folder", command=lambda: self._run_selected_function("Open Results Folder")).pack(anchor="w", pady=(12, 0))

    def _add_field_row_to_parent(self, parent, label_text, variable, browse_func=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text=f"{label_text}:", width=22, anchor="w").pack(side="left", padx=(0, 8))
        entry = ttk.Entry(row, textvariable=variable, width=80)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        if browse_func is not None:
            ttk.Button(row, text="Browse", command=browse_func).pack(side="left")

    def _add_checkbox_row_to_parent(self, parent, label_text, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)
        ttk.Checkbutton(row, text=label_text, variable=variable).pack(anchor="w")

    def _add_info_label_to_parent(self, parent, text):
        label = ttk.Label(parent, text=text, foreground="#325d8c", wraplength=800, justify="left")
        label.pack(anchor="w", pady=(12, 4))

    def _choose_auto_recording_dir(self):
        folder = filedialog.askdirectory(title="Choose recording folder", initialdir=self.auto_recording_dir_var.get())
        if folder:
            self.auto_recording_dir_var.set(folder)

    def _choose_auto_db_path(self):
        file = filedialog.asksaveasfilename(
            title="Choose SQLite database path",
            defaultextension=".sqlite3",
            filetypes=[("SQLite database", "*.sqlite3"), ("All files", "*.*")],
            initialdir=str(Path(self.auto_db_path_var.get()).parent),
        )
        if file:
            self.auto_db_path_var.set(file)

    def _choose_auto_output_dir(self):
        folder = filedialog.askdirectory(title="Choose detection output folder", initialdir=self.auto_output_dir_var.get())
        if folder:
            self.auto_output_dir_var.set(folder)

    def _choose_dataset(self):
        file = filedialog.askopenfilename(
            title="Choose dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(APP_DIR / "data" / "training"),
        )
        if file:
            self.dataset_var.set(file)

    def _choose_model_path(self):
        file = filedialog.asksaveasfilename(
            title="Choose model path",
            defaultextension=".joblib",
            filetypes=[(".joblib files", "*.joblib"), ("All files", "*.*")],
            initialdir=str(APP_DIR / "models"),
        )
        if file:
            self.model_var.set(file)

    def _choose_input(self):
        file = filedialog.askopenfilename(
            title="Choose capture input",
            filetypes=[("Numpy arrays", "*.npy"), ("Binary IQ", "*.bin *.iq *.dat *.raw"), ("All files", "*.*")],
            initialdir=str(APP_DIR / "data" / "raw"),
        )
        if file:
            self.input_var.set(file)

    def _choose_output_dir(self):
        folder = filedialog.askdirectory(title="Choose output folder", initialdir=str(APP_DIR / "data" / "results"))
        if folder:
            self.output_dir_var.set(folder)

    def _choose_db_path(self):
        file = filedialog.asksaveasfilename(
            title="Choose SQLite database path",
            defaultextension=".sqlite3",
            filetypes=[("SQLite database", "*.sqlite3"), ("All files", "*.*")],
            initialdir=str(APP_DIR / "data" / "results"),
        )
        if file:
            self.db_path_var.set(file)

    def _run_selected_function(self, selected):
        if selected == "Open Results Folder":
            self._open_results_folder()
        elif selected == "Train Model":
            self._train_model()
        elif selected == "Run Detection":
            self._run_detection()
        elif selected == "Auto Capture":
            self._run_auto_capture()

    def _run_auto_capture(self):
        try:
            frequency = float(self.auto_frequency_var.get())
            sample_rate = float(self.auto_sample_rate_var.get())
            duration = float(self.auto_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Frequency, sample rate, and duration must be numbers.")
            return

        satellite = self.auto_satellite_var.get().strip()
        if not satellite:
            messagebox.showerror("Missing satellite name", "Enter a satellite/target name.")
            return

        recording_dir = Path(self.auto_recording_dir_var.get()).expanduser()
        recording_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(self.auto_db_path_var.get()).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(self.auto_output_dir_var.get()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--satellite", satellite,
            "--frequency", str(frequency),
            "--sample-rate", str(sample_rate),
            "--duration", str(duration),
            "--gain", self.auto_gain_var.get().strip() or "auto",
            "--output-dir", str(recording_dir),
            "--db", str(db_path),
            "--detection-output-dir", str(output_dir),
        ]
        norad_id = self.auto_norad_id_var.get().strip()
        if norad_id:
            cmd.extend(["--norad-id", norad_id])
        if self.auto_simulate_var.get():
            cmd.append("--simulate")
        if self.auto_save_image_var.get():
            cmd.append("--save-image")
        if self.auto_no_ml_var.get():
            cmd.append("--no-ml")

        # Keep the Run Detection / Open Results Folder tabs pointed at the
        # same output/db this run just used, so the Results and Image
        # Preview panels below pick it up.
        self.output_dir_var.set(str(output_dir))
        self.db_path_var.set(str(db_path))

        self._run_command(cmd, title="Auto capture", cwd=PROJECT_ROOT)

    def _train_model(self):
        dataset = Path(self.dataset_var.get()).expanduser()
        model_path = Path(self.model_var.get()).expanduser()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not dataset.exists():
            messagebox.showerror("Dataset missing", f"The dataset does not exist:\n{dataset}")
            return

        cmd = [sys.executable, "train_model.py", "--dataset", str(dataset), "--output", str(model_path)]
        if self.allow_synthetic_var.get():
            cmd.append("--allow-synthetic")
        self._run_command(cmd, title="Training model")

    def _run_detection(self):
        capture = Path(self.input_var.get()).expanduser()
        output_dir = Path(self.output_dir_var.get()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(self.model_var.get()).expanduser()
        db_path = Path(self.db_path_var.get()).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if not capture.exists():
            messagebox.showerror("Capture missing", f"The capture does not exist:\n{capture}")
            return

        sample_rate = self.sample_rate_var.get().strip() or str(DEFAULT_SAMPLE_RATE_HZ)
        center_freq = self.center_freq_var.get().strip() or str(DEFAULT_CENTER_FREQ_HZ)
        nperseg = self.nperseg_var.get().strip() or str(DEFAULT_NPERSEG)
        noverlap = self.noverlap_var.get().strip() or str(DEFAULT_NOVERLAP)
        binary_dtype = self.binary_dtype_var.get().strip() or "complex64"
        snr_threshold = self.snr_threshold_db_var.get().strip() or str(DEFAULT_SNR_THRESHOLD_DB)
        min_valid_ratio = self.min_valid_ratio_var.get().strip() or str(DEFAULT_MIN_VALID_RATIO)
        min_drift_hz = self.min_drift_hz_var.get().strip() or str(DEFAULT_MIN_DRIFT_HZ)
        max_smoothness_hz = self.max_smoothness_hz_var.get().strip() or str(DEFAULT_MAX_SMOOTHNESS_HZ)
        chunk_size = self.chunk_size_var.get().strip() or "0"
        chunk_overlap = self.chunk_overlap_var.get().strip() or "0"

        cmd = [
            sys.executable,
            "src/main.py",
            "--input", str(capture),
            "--output", str(output_dir),
            "--db", str(db_path),
            "--sample-rate", sample_rate,
            "--center-freq", center_freq,
            "--nperseg", nperseg,
            "--noverlap", noverlap,
            "--binary-dtype", binary_dtype,
            "--snr-threshold-db", snr_threshold,
            "--min-valid-ratio", min_valid_ratio,
            "--min-drift-hz", min_drift_hz,
            "--max-smoothness-hz", max_smoothness_hz,
            "--ml-model", str(model_path),
        ]
        if self.save_image_var.get():
            cmd.append("--save-image")
            try:
                chunk_size_int = int(chunk_size)
                overlap_int = int(chunk_overlap)
                if chunk_size_int > 0:
                    cmd.extend(["--image-chunk-size", str(chunk_size_int), "--image-chunk-overlap", str(overlap_int)])
            except ValueError:
                messagebox.showerror("Invalid image chunk settings", "Chunk size and overlap must be integers.")
                return
        if self.no_ml_var.get():
            cmd.append("--no-ml")

        self._run_command(cmd, title="Running detection", cwd=APP_DIR)

    def _open_results_folder(self):
        try:
            base_path = Path(self.output_dir_var.get()).expanduser()
            base_path.mkdir(parents=True, exist_ok=True)
            latest_session = self._latest_result_folder()
            path = latest_session if latest_session.exists() and any(latest_session.rglob("*_summary.json")) else base_path
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Folder error", f"Unable to open results folder: {exc}")

    def _run_command(self, cmd, title: str, cwd: Path = None):
        self.status_var.set(f"{title} running...")
        self.progress_text_var.set(f"Working: {title}")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(14)

        def worker():
            try:
                result = subprocess.run(cmd, cwd=str(cwd or APP_DIR), capture_output=True, text=True)
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                combined = "\n".join(part for part in [stdout, stderr] if part)
                if result.returncode == 0:
                    self.status_var.set(f"{title} complete")
                else:
                    self.status_var.set(f"{title} failed")
                self.root.after(0, lambda: self._show_command_output(combined, result.returncode))
            except Exception as exc:
                self.root.after(0, lambda: self._show_command_output(f"Error: {exc}", 1))
                self.root.after(0, lambda: self.status_var.set(f"{title} failed"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_command_output(self, output: str, return_code: int):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_text_var.set("Complete")

        self.results_box.config(state="normal")
        self.results_box.delete("1.0", "end")
        if output:
            self.results_box.insert("end", output)
        else:
            self.results_box.insert("end", "The command completed without any text output.")
        self.results_box.config(state="disabled")

        if return_code == 0:
            self._refresh_result_panel()
            self._refresh_image_preview()
            self._open_results_window()

    def _open_results_window(self):
        output_dir = self._latest_result_folder()
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_files = sorted(output_dir.rglob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        payload = None
        title = "Detection results"
        if summary_files:
            chosen = summary_files[0]
            title = chosen.name
            try:
                with open(chosen, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = None

        result_text = "No result file found yet."
        if payload is not None:
            if "evaluation_metrics" in payload or "feature_importance" in payload:
                result_text = format_training_summary(payload)
            else:
                result_text = format_detection_summary(payload)
        elif self.results_box.get("1.0", "end").strip():
            result_text = self.results_box.get("1.0", "end").strip()

        window = Toplevel(self.root)
        window.title(f"{title} - Results")
        window.geometry("980x700")
        window.minsize(760, 520)

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        results_tab = ttk.Frame(notebook)
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(0, weight=1)
        notebook.add(results_tab, text="Results")

        result_box = ScrolledText(results_tab, wrap="word", state="disabled", height=28)
        result_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        result_box.config(state="normal")
        result_box.delete("1.0", "end")
        result_box.insert("end", result_text)
        result_box.config(state="disabled")

        png_files = sorted(output_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime)
        if png_files:
            image_tab = ttk.Frame(notebook)
            image_tab.grid_columnconfigure(0, weight=1)
            image_tab.grid_rowconfigure(0, weight=1)
            notebook.add(image_tab, text="Image Preview")

            nav_bar = ttk.Frame(image_tab)
            nav_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
            prev_button = ttk.Button(nav_bar, text="Previous", command=lambda: self._show_sequence_image(png_files, -1, image_label, counter_var, prev_button, next_button))
            prev_button.pack(side="left")
            counter_var = StringVar(value="0 / 0")
            ttk.Label(nav_bar, textvariable=counter_var, width=16, anchor="center").pack(side="left", padx=10)
            next_button = ttk.Button(nav_bar, text="Next", command=lambda: self._show_sequence_image(png_files, 1, image_label, counter_var, prev_button, next_button))
            next_button.pack(side="left")

            image_canvas = ttk.Frame(image_tab)
            image_canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
            image_canvas.grid_columnconfigure(0, weight=1)
            image_canvas.grid_rowconfigure(0, weight=1)

            image_label = ttk.Label(image_canvas, text="Loading image preview...", anchor="center")
            image_label.grid(row=0, column=0, sticky="nsew")

            self._show_sequence_image(png_files, 0, image_label, counter_var, prev_button, next_button)

        window.transient(self.root)
        window.grab_set()

    def _latest_result_folder(self) -> Path:
        base_dir = Path(self.output_dir_var.get()).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        session_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
        if session_dirs:
            return max(session_dirs, key=lambda p: p.stat().st_mtime)
        return base_dir

    def _show_sequence_image(self, png_files, step, image_label, counter_var, prev_button, next_button):
        if not png_files:
            counter_var.set("0 / 0")
            image_label.configure(text="No PNG preview available yet.")
            image_label.image = None
            return

        if not hasattr(self, "current_image_index"):
            self.current_image_index = 0

        if step == 0:
            self.current_image_index = 0
        else:
            self.current_image_index = max(0, min(self.current_image_index + step, len(png_files) - 1))

        image_path = png_files[self.current_image_index]
        counter_var.set(f"{self.current_image_index + 1} / {len(png_files)}")
        prev_button.configure(state="normal" if self.current_image_index > 0 else "disabled")
        next_button.configure(state="normal" if self.current_image_index < len(png_files) - 1 else "disabled")

        if Image is not None and ImageTk is not None:
            try:
                with Image.open(image_path) as image:
                    image.thumbnail((860, 480))
                    photo = ImageTk.PhotoImage(image)
                image_label.configure(image=photo, text="")
                image_label.image = photo
                return
            except Exception:
                pass

        image_label.configure(
            text=f"Image {self.current_image_index + 1} of {len(png_files)}:\n{image_path.name}\n\nOutput folder: {image_path.parent.name}\n\nOpen it from the output folder or use a regular image viewer.",
        )
        image_label.image = None

    def _show_previous_image(self):
        if not self.image_series:
            return
        self._show_sequence_image(self.image_series, -1, self.image_label, self.image_counter_var, self.image_prev_button, self.image_next_button)

    def _show_next_image(self):
        if not self.image_series:
            return
        self._show_sequence_image(self.image_series, 1, self.image_label, self.image_counter_var, self.image_prev_button, self.image_next_button)

    def _latest_png_files(self):
        base_dir = Path(self.output_dir_var.get()).expanduser()
        if not base_dir.exists():
            return []
        session_dir = self._latest_result_folder()
        return sorted(session_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime)

    def _refresh_result_panel(self):
        output_dir = self._latest_result_folder()
        summary_files = sorted(output_dir.rglob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        payload = None
        title = "No result available"
        if summary_files:
            chosen = summary_files[0]
            title = chosen.name
            try:
                with open(chosen, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = None

        training_model = Path(self.model_var.get()).expanduser()
        if payload is None and training_model.exists() and training_model.suffix == ".joblib":
            metadata_path = training_model.with_suffix(".json")
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    title = metadata_path.name
                except Exception:
                    payload = None

        if payload is None:
            content = "No result file found yet.\nRun Auto Capture, Run Detection, or Train Model to generate results."
        elif "evaluation_metrics" in payload or "feature_importance" in payload:
            content = format_training_summary(payload)
        else:
            content = format_detection_summary(payload)

        self.results_box.config(state="normal")
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", f"Result folder: {output_dir.name}\n\nResult: {title}\n\n{content}\n\n")
        if payload is not None:
            self.results_box.insert("end", "Raw JSON data:\n")
            self.results_box.insert("end", json.dumps(payload, indent=2, sort_keys=True))
        self.results_box.config(state="disabled")

    def _refresh_image_preview(self):
        self.image_series = self._latest_png_files()
        if not self.image_series:
            self.image_counter_var.set("0 / 0")
            self.image_prev_button.configure(state="disabled")
            self.image_next_button.configure(state="disabled")
            self.image_label.configure(text="No PNG preview available yet.")
            self.image_label.image = None
            return

        self.current_image_index = max(0, min(self.current_image_index, len(self.image_series) - 1))
        self._show_sequence_image(
            self.image_series, 0, self.image_label, self.image_counter_var, self.image_prev_button, self.image_next_button
        )


def main():
    root = Tk()
    SDRDopplerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
