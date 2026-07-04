#!/usr/bin/env python3
"""
Tkinter GUI for quality-based review. Sorted by laplacian_blur (worst first).
Same controls as cluster review: P=pass page, T=trash page, click+1/2 for individual.

Reads:  config.yaml -> paths.images_dir, quality.*, review.*
Reads:  data/output/03_quality.csv, data/output/combined_labels.json
Writes: updates combined_labels.json in-place (adds human_label field)
"""

import csv
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import load_json, save_json

GRID_COLS = 5
GRID_ROWS = 5
CELLS_PER_PAGE = GRID_COLS * GRID_ROWS
THUMB_SIZE = (150, 80)
BORDER = 3

COLOR_BG = "#1e1e1e"
COLOR_UNREVIEWED = "#555"
COLOR_PASS = "#00ff88"
COLOR_TRASH = "#ff4444"
COLOR_SELECTED = "#ffaa00"
COLOR_TEXT = "#ccc"


class QualityReviewGUI:
    def __init__(self, root, images_dir, labels_path, quality_csv):
        self.root = root
        self.images_dir = Path(images_dir)
        self.labels_path = Path(labels_path)
        self.labels = load_json(labels_path)
        self.quality_data = self._load_quality(quality_csv)
        self.page_offset = 0
        self.selected_cell = None

        self._setup_ui()
        self._render_page()

    def _load_quality(self, csv_path):
        data = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        try:
            data.sort(key=lambda r: float(r.get("laplacian_blur", 0)))
        except (ValueError, TypeError):
            pass
        return data

    def _setup_ui(self):
        self.root.title("Quality Review GUI")
        self.root.configure(bg=COLOR_BG)

        self.info_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.title_label = tk.Label(
            self.info_frame, text="Quality Review — Worst Images First",
            fg=COLOR_TEXT, bg=COLOR_BG, font=tkfont.Font(size=14, weight="bold"))
        self.title_label.pack(side=tk.LEFT)

        self.progress_label = tk.Label(self.info_frame, text="", fg=COLOR_TEXT, bg=COLOR_BG)
        self.progress_label.pack(side=tk.RIGHT)

        self.grid_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.grid_frame.pack(padx=10, pady=5)

        self.cells = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cell_frame = tk.Frame(
                    self.grid_frame, bg=COLOR_UNREVIEWED,
                    highlightbackground=COLOR_UNREVIEWED,
                    highlightthickness=BORDER,
                    width=THUMB_SIZE[0], height=THUMB_SIZE[1])
                cell_frame.grid(row=row, column=col, padx=2, pady=2)
                cell_frame.grid_propagate(False)

                img_label = tk.Label(cell_frame, bg=COLOR_UNREVIEWED)
                img_label.pack(fill=tk.BOTH, expand=True)

                cell_idx = row * GRID_COLS + col
                img_label.bind("<Button-1>", lambda e, c=cell_idx: self._on_select(c))
                cell_frame.bind("<Button-1>", lambda e, c=cell_idx: self._on_select(c))

                self.cells.append({"frame": cell_frame, "label": img_label, "row": None})

        self.ctrl_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(self.ctrl_frame,
                 text="P=Pass Page  T=Trash Page  Left/Right=Navigate  Click+1=Pass  Click+2=Trash",
                 fg=COLOR_TEXT, bg=COLOR_BG).pack()

        self.root.bind("<Key-p>", lambda e: self._batch_label("pass"))
        self.root.bind("<Key-t>", lambda e: self._batch_label("trash"))
        self.root.bind("<Key-1>", lambda e: self._individual_label("pass"))
        self.root.bind("<Key-2>", lambda e: self._individual_label("trash"))
        self.root.bind("<Left>", lambda e: self._prev_page())
        self.root.bind("<Right>", lambda e: self._next_page())

    def _render_page(self):
        start = self.page_offset * CELLS_PER_PAGE
        page_data = self.quality_data[start : start + CELLS_PER_PAGE]
        total_pages = max(1, (len(self.quality_data) - 1) // CELLS_PER_PAGE + 1)
        self.progress_label.config(
            text=f"{start + 1}-{min(start + CELLS_PER_PAGE, len(self.quality_data))} / {len(self.quality_data)}  Page {self.page_offset + 1}/{total_pages}")

        for i, cell in enumerate(self.cells):
            if i < len(page_data):
                row = page_data[i]
                stem = os.path.splitext(row.get("image_file", ""))[0]
                cell["row"] = row
                img_path = self.images_dir / row.get("image_file", "")
                color = COLOR_UNREVIEWED

                if stem and stem in self.labels:
                    existing = self.labels[stem].get("human_label")
                    if existing == "pass":
                        color = COLOR_PASS
                    elif existing == "trash":
                        color = COLOR_TRASH

                cell["frame"].configure(highlightbackground=color)
                cell["label"].configure(bg=color)

                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        cell["label"].configure(image=photo)
                        cell["label"].image = photo
                    except Exception:
                        cell["label"].configure(image="", text="?")
            else:
                cell["row"] = None
                cell["frame"].configure(highlightbackground=COLOR_BG)
                cell["label"].configure(image="", bg=COLOR_BG)

    def _on_select(self, cell_idx):
        self.selected_cell = cell_idx

    def _batch_label(self, label):
        start = self.page_offset * CELLS_PER_PAGE
        page_data = self.quality_data[start : start + CELLS_PER_PAGE]
        count = 0
        for row in page_data:
            stem = os.path.splitext(row.get("image_file", ""))[0]
            if stem and stem in self.labels:
                self.labels[stem]["human_label"] = label
                count += 1
        save_json(self.labels_path, self.labels)
        self._render_page()
        print(f"  Page marked as '{label}' — {count} images")

    def _individual_label(self, label):
        if self.selected_cell is None:
            return
        cell = self.cells[self.selected_cell]
        row = cell["row"]
        if row is None:
            return
        stem = os.path.splitext(row.get("image_file", ""))[0]
        if stem and stem in self.labels:
            self.labels[stem]["human_label"] = label
        save_json(self.labels_path, self.labels)
        print(f"  {row.get('image_file')}: marked '{label}'")

    def _next_page(self):
        max_page = (len(self.quality_data) - 1) // CELLS_PER_PAGE
        if self.page_offset < max_page:
            self.page_offset += 1
            self._render_page()

    def _prev_page(self):
        if self.page_offset > 0:
            self.page_offset -= 1
            self._render_page()


def main():
    cfg = load_config()
    images_dir = cfg["paths"]["images_dir"]
    output_dir = cfg["paths"]["output_dir"]
    labels_path = cfg["review"]["combined_labels_file"]
    quality_csv = output_dir / "03_quality.csv"

    root_dir = Path(__file__).resolve().parent.parent
    if not os.path.isabs(labels_path):
        labels_path = str(root_dir / labels_path)

    root = tk.Tk()
    root.geometry("900x700")
    QualityReviewGUI(root, images_dir, labels_path, quality_csv)
    root.mainloop()


if __name__ == "__main__":
    main()
