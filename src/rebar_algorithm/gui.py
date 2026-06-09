"""Minimal Tkinter GUI for selecting SAM prompt points and viewing results."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from loguru import logger
from PIL import Image, ImageTk

from .app_api import PipelineRunResult, find_project_image, run_pipeline_from_points
from .config import get_rebar_config


DETECTOR_CHOICES = ("mask-grid", "yolo")
MAX_PREVIEW_SIZE = (1100, 820)


class RebarGui(tk.Tk):
    """Small desktop wrapper around the existing rebar pipeline."""

    def __init__(self):
        super().__init__()
        self.title("Rebar Detection")
        self.minsize(1120, 760)

        self.project_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.detector_var = tk.StringVar(value="mask-grid")
        self.plane_enabled_var = tk.BooleanVar(value=True)
        self.plane_threshold_var = tk.StringVar(value="0.03")
        self.yolo_url_var = tk.StringVar()
        self.use_existing_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

        self.points: list[tuple[int, int]] = []
        self.input_image_path: Optional[Path] = None
        self.input_image: Optional[Image.Image] = None
        self.input_preview_size = (0, 0)
        self.input_scale = 1.0
        self.input_photo: Optional[ImageTk.PhotoImage] = None
        self.result_photo: Optional[ImageTk.PhotoImage] = None
        self.last_result: Optional[PipelineRunResult] = None

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: Optional[threading.Thread] = None

        self._load_defaults()
        self._build_ui()
        self.after(100, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _load_defaults(self) -> None:
        try:
            cfg = get_rebar_config()
            self.plane_enabled_var.set(cfg.is_plane_extraction_enabled())
            self.plane_threshold_var.set(str(cfg.get_plane_distance_threshold()))
            self.yolo_url_var.set(cfg.get_server_url())
        except Exception:
            self.plane_threshold_var.set("0.03")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(paned, width=330)
        viewer = ttk.Frame(paned)
        paned.add(sidebar, weight=0)
        paned.add(viewer, weight=1)

        self._build_sidebar(sidebar)
        self._build_viewer(viewer)

        status = ttk.Label(root, textvariable=self.status_var, anchor=tk.W)
        status.pack(fill=tk.X, pady=(8, 0))

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        project_box = ttk.LabelFrame(parent, text="Project")
        project_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        project_box.columnconfigure(0, weight=1)

        ttk.Entry(project_box, textvariable=self.project_var).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4)
        )
        ttk.Button(project_box, text="Browse", command=self._browse_project).grid(
            row=1, column=0, sticky="ew", padx=(8, 4), pady=(0, 8)
        )
        ttk.Button(project_box, text="Load", command=self._load_project).grid(
            row=1, column=1, sticky="ew", padx=(4, 8), pady=(0, 8)
        )

        output_box = ttk.LabelFrame(parent, text="Output")
        output_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        output_box.columnconfigure(0, weight=1)
        ttk.Entry(output_box, textvariable=self.output_var).grid(
            row=0, column=0, sticky="ew", padx=8, pady=(8, 4)
        )
        ttk.Button(output_box, text="Browse", command=self._browse_output).grid(
            row=0, column=1, sticky="ew", padx=(0, 8), pady=(8, 4)
        )
        self.open_output_button = ttk.Button(
            output_box, text="Open Output Folder", command=self._open_output_folder
        )
        self.open_output_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        run_box = ttk.LabelFrame(parent, text="Run")
        run_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        run_box.columnconfigure(1, weight=1)

        ttk.Label(run_box, text="Detector").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Combobox(
            run_box,
            textvariable=self.detector_var,
            values=DETECTOR_CHOICES,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=(8, 4))

        ttk.Checkbutton(
            run_box,
            text="Plane extraction",
            variable=self.plane_enabled_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        ttk.Label(run_box, text="Threshold").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(run_box, textvariable=self.plane_threshold_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=(4, 8), pady=4
        )

        ttk.Label(run_box, text="YOLO URL").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(run_box, textvariable=self.yolo_url_var).grid(
            row=3, column=1, sticky="ew", padx=(4, 8), pady=4
        )

        ttk.Checkbutton(
            run_box,
            text="Reuse YOLO annotations",
            variable=self.use_existing_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        self.run_button = ttk.Button(run_box, text="Run Pipeline", command=self._start_run)
        self.run_button.grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 8))

        points_box = ttk.LabelFrame(parent, text="Prompt Points")
        points_box.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        parent.rowconfigure(3, weight=1)
        points_box.columnconfigure(0, weight=1)
        points_box.rowconfigure(0, weight=1)

        self.points_list = tk.Listbox(points_box, height=8, activestyle="dotbox")
        self.points_list.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=(8, 4))
        ttk.Button(points_box, text="Undo", command=self._undo_point).grid(
            row=1, column=0, sticky="ew", padx=(8, 4), pady=(0, 8)
        )
        ttk.Button(points_box, text="Clear", command=self._clear_points).grid(
            row=1, column=1, sticky="ew", padx=(4, 8), pady=(0, 8)
        )

        log_box = ttk.LabelFrame(parent, text="Status Log")
        log_box.grid(row=4, column=0, sticky="nsew")
        parent.rowconfigure(4, weight=1)
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_box, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        log_scroll = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.bind("<Return>", lambda _event: self._load_project())

    def _build_viewer(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        input_tab = ttk.Frame(self.notebook)
        result_tab = ttk.Frame(self.notebook)
        self.notebook.add(input_tab, text="Input")
        self.notebook.add(result_tab, text="Result")

        input_tab.columnconfigure(0, weight=1)
        input_tab.rowconfigure(0, weight=1)
        result_tab.columnconfigure(0, weight=1)
        result_tab.rowconfigure(0, weight=1)

        self.input_canvas = tk.Canvas(input_tab, background="#1f2328", highlightthickness=0)
        self.input_canvas.grid(row=0, column=0, sticky="nsew")
        input_y_scroll = ttk.Scrollbar(input_tab, orient=tk.VERTICAL, command=self.input_canvas.yview)
        input_y_scroll.grid(row=0, column=1, sticky="ns")
        input_x_scroll = ttk.Scrollbar(input_tab, orient=tk.HORIZONTAL, command=self.input_canvas.xview)
        input_x_scroll.grid(row=1, column=0, sticky="ew")
        self.input_canvas.configure(xscrollcommand=input_x_scroll.set, yscrollcommand=input_y_scroll.set)
        self.input_canvas.bind("<Button-1>", self._add_point_from_event)

        self.result_canvas = tk.Canvas(result_tab, background="#1f2328", highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        result_y_scroll = ttk.Scrollbar(result_tab, orient=tk.VERTICAL, command=self.result_canvas.yview)
        result_y_scroll.grid(row=0, column=1, sticky="ns")
        result_x_scroll = ttk.Scrollbar(result_tab, orient=tk.HORIZONTAL, command=self.result_canvas.xview)
        result_x_scroll.grid(row=1, column=0, sticky="ew")
        self.result_canvas.configure(xscrollcommand=result_x_scroll.set, yscrollcommand=result_y_scroll.set)

    # ------------------------------------------------------------------
    # Project and image handling
    # ------------------------------------------------------------------
    def _browse_project(self) -> None:
        path = filedialog.askdirectory(title="Select project folder")
        if path:
            self.project_var.set(path)
            self._load_project()

    def _browse_output(self) -> None:
        initial_dir = self.output_var.get() or self.project_var.get() or str(Path.home())
        path = filedialog.askdirectory(title="Select output folder", initialdir=initial_dir)
        if path:
            self.output_var.set(path)

    def _load_project(self) -> None:
        project_text = self.project_var.get().strip()
        if not project_text:
            return
        project_path = Path(project_text).expanduser()
        if not project_path.is_dir():
            messagebox.showerror("Project Not Found", f"Project folder not found:\n{project_path}")
            return

        try:
            self.input_image_path = find_project_image(project_path)
            self.input_image = Image.open(self.input_image_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image Not Found", str(exc))
            return

        if not self.output_var.get().strip():
            self.output_var.set(str(project_path / "rebar_output"))
        self.points.clear()
        self._refresh_point_list()
        self._render_input_image()
        self._clear_result_image()
        self._set_status(f"Loaded {self.input_image_path.name}")
        self._append_log(f"Loaded project: {project_path}\n")

    def _render_input_image(self) -> None:
        if self.input_image is None:
            self.input_canvas.delete("all")
            return

        preview = self.input_image.copy()
        preview.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.input_preview_size = preview.size
        self.input_scale = preview.size[0] / self.input_image.size[0]
        self.input_photo = ImageTk.PhotoImage(preview)

        self.input_canvas.delete("all")
        self.input_canvas.config(scrollregion=(0, 0, preview.size[0], preview.size[1]))
        self.input_canvas.create_image(0, 0, image=self.input_photo, anchor=tk.NW, tags=("image",))
        self._draw_points()

    def _render_result_image(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        preview = image.copy()
        preview.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.result_photo = ImageTk.PhotoImage(preview)

        self.result_canvas.delete("all")
        self.result_canvas.config(scrollregion=(0, 0, preview.size[0], preview.size[1]))
        self.result_canvas.create_image(0, 0, image=self.result_photo, anchor=tk.NW)
        self.notebook.select(1)

    def _clear_result_image(self) -> None:
        self.result_canvas.delete("all")
        self.last_result = None

    # ------------------------------------------------------------------
    # Point editing
    # ------------------------------------------------------------------
    def _add_point_from_event(self, event: tk.Event) -> None:
        if self.input_image is None:
            return
        canvas_x = self.input_canvas.canvasx(event.x)
        canvas_y = self.input_canvas.canvasy(event.y)
        x = int(round(canvas_x / self.input_scale))
        y = int(round(canvas_y / self.input_scale))
        width, height = self.input_image.size
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        self.points.append((x, y))
        self._refresh_point_list()
        self._draw_points()

    def _undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._refresh_point_list()
            self._draw_points()

    def _clear_points(self) -> None:
        self.points.clear()
        self._refresh_point_list()
        self._draw_points()

    def _refresh_point_list(self) -> None:
        self.points_list.delete(0, tk.END)
        for index, (x, y) in enumerate(self.points, start=1):
            self.points_list.insert(tk.END, f"{index}. {x}, {y}")

    def _draw_points(self) -> None:
        self.input_canvas.delete("point")
        if self.input_image is None:
            return
        radius = 6
        for index, (x, y) in enumerate(self.points, start=1):
            dx = x * self.input_scale
            dy = y * self.input_scale
            self.input_canvas.create_oval(
                dx - radius,
                dy - radius,
                dx + radius,
                dy + radius,
                fill="#ffd43b",
                outline="#111111",
                width=2,
                tags=("point",),
            )
            self.input_canvas.create_text(
                dx + 12,
                dy - 12,
                text=str(index),
                fill="#ffd43b",
                anchor=tk.W,
                tags=("point",),
            )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        try:
            project_path = self._validated_project_path()
            output_path = self._validated_output_path(project_path)
            threshold = self._validated_threshold()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        points = list(self.points)
        detector = self.detector_var.get()
        yolo_url = self.yolo_url_var.get().strip() or None
        use_existing = self.use_existing_var.get()
        enable_plane = self.plane_enabled_var.get()

        self.run_button.config(state=tk.DISABLED)
        self._clear_result_image()
        self._set_status("Running pipeline...")
        self._append_log("\nRunning pipeline...\n")

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(project_path, output_path, points, detector, yolo_url, use_existing, enable_plane, threshold),
            daemon=True,
        )
        self.worker.start()

    def _validated_project_path(self) -> Path:
        project_text = self.project_var.get().strip()
        if not project_text:
            raise ValueError("Select a project folder first.")
        project_path = Path(project_text).expanduser()
        if not project_path.is_dir():
            raise ValueError(f"Project folder not found:\n{project_path}")
        if not self.points:
            raise ValueError("Add at least one prompt point before running.")
        return project_path

    def _validated_output_path(self, project_path: Path) -> Path:
        output_text = self.output_var.get().strip()
        output_path = Path(output_text).expanduser() if output_text else project_path / "rebar_output"
        self.output_var.set(str(output_path))
        return output_path

    def _validated_threshold(self) -> Optional[float]:
        if not self.plane_enabled_var.get():
            return None
        threshold_text = self.plane_threshold_var.get().strip()
        if not threshold_text:
            return None
        try:
            return float(threshold_text)
        except ValueError:
            raise ValueError("Plane threshold must be a number.")

    def _run_worker(
        self,
        project_path: Path,
        output_path: Path,
        points: list[tuple[int, int]],
        detector: str,
        yolo_url: Optional[str],
        use_existing: bool,
        enable_plane: bool,
        threshold: Optional[float],
    ) -> None:
        handler_id = logger.add(
            lambda message: self.events.put(("log", str(message))),
            level="INFO",
            format="{message}",
        )
        try:
            result = run_pipeline_from_points(
                project_path=project_path,
                points=points,
                output_path=output_path,
                detector=detector,
                yolo_url=yolo_url,
                use_existing_annotations=use_existing,
                enable_plane_extraction=enable_plane,
                plane_distance_threshold=threshold,
            )
            self.events.put(("result", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))
        finally:
            logger.remove(handler_id)
            self.events.put(("done", None))

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "result":
                result = payload
                if isinstance(result, PipelineRunResult):
                    self.last_result = result
                    self._render_result_image(result.final_image_path)
                    self._set_status(f"Complete: {result.final_image_path}")
                    self._append_log(f"\nResult image: {result.final_image_path}\n")
                    if result.analysis_json_path:
                        self._append_log(f"Analysis JSON: {result.analysis_json_path}\n")
            elif kind == "error":
                text = str(payload)
                self._append_log(text + "\n")
                self._set_status("Failed")
                messagebox.showerror("Pipeline Failed", self._short_error(text))
            elif kind == "done":
                self.run_button.config(state=tk.NORMAL)

        self.after(100, self._drain_events)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _open_output_folder(self) -> None:
        path_text = self.output_var.get().strip()
        if self.last_result:
            path = self.last_result.output_path
        elif path_text:
            path = Path(path_text).expanduser()
        elif self.project_var.get().strip():
            path = Path(self.project_var.get().strip()).expanduser() / "rebar_output"
        else:
            return

        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])

    @staticmethod
    def _short_error(trace: str) -> str:
        lines = [line for line in trace.strip().splitlines() if line.strip()]
        return lines[-1] if lines else "Unknown error"

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            ok = messagebox.askokcancel("Quit", "The pipeline is still running. Quit anyway?")
            if not ok:
                return
        self.destroy()


def main() -> None:
    app = RebarGui()
    app.mainloop()


if __name__ == "__main__":
    main()
