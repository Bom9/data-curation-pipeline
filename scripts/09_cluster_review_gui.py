#!/usr/bin/env python3
"""
Tkinter GUI for batch review of clustered plate images.

Keyboard controls: P=pass page, T=trash page, Click+1=pass, Click+2=trash, arrows=navigate.
Individual labels take priority over batch labels.

Reads:  config.yaml -> paths.images_dir, review.*
Reads:  data/output/cluster_labels.json, data/output/combined_labels.json
Writes: updates combined_labels.json in-place (adds human_label field)
"""

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


class ClusterReviewGUI:
    def __init__(self, root, images_dir, labels_path, clusters_path):
        self.root = root
        self.images_dir = Path(images_dir)
        self.labels_path = Path(labels_path)
        self.clusters_path = Path(clusters_path)
        self.labels = load_json(labels_path)
        self.clusters = load_json(clusters_path)

        self.cluster_to_images = {}
        for stem, cluster_id in self.clusters.items():
            cid = str(cluster_id)
            self.cluster_to_images.setdefault(cid, []).append(stem)

        self.cluster_ids = sorted(
            self.cluster_to_images.keys(),
            key=lambda x: len(self.cluster_to_images[x]),
            reverse=True,
        )
        self.cluster_idx = 0
        self.page_offset = 0
        self.selected_cell = None
        self._setup_ui()
        self._load_cluster(0)

    def _setup_ui(self):
        self.root.title("Cluster Review GUI")
        self.root.configure(bg=COLOR_BG)

        self.info_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.cluster_label = tk.Label(
            self.info_frame, text="", fg=COLOR_TEXT, bg=COLOR_BG,
            font=tkfont.Font(size=14, weight="bold"))
        self.cluster_label.pack(side=tk.LEFT)

        self.progress_label = tk.Label(
            self.info_frame, text="", fg=COLOR_TEXT, bg=COLOR_BG)
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

                self.cells.append({"frame": cell_frame, "label": img_label, "stem": None})

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

    def _load_cluster(self, idx):
        if idx < 0 or idx >= len(self.cluster_ids):
            return
        self.cluster_idx = idx
        self.page_offset = 0
        self._render_page()

    def _render_page(self):
        cid = self.cluster_ids[self.cluster_idx]
        stems = self.cluster_to_images[cid]
        start = self.page_offset * CELLS_PER_PAGE
        page_stems = stems[start : start + CELLS_PER_PAGE]

        self.cluster_label.config(
            text=f"Cluster {cid}  ({len(stems)} images)  [{self.cluster_idx + 1}/{len(self.cluster_ids)}]")
        total_pages = max(1, (len(stems) - 1) // CELLS_PER_PAGE + 1)
        self.progress_label.config(text=f"Page {self.page_offset + 1}/{total_pages}")

        for i, cell in enumerate(self.cells):
            if i < len(page_stems):
                stem = page_stems[i]
                cell["stem"] = stem
                img_path = self.images_dir / f"{stem}.jpg"
                color = COLOR_UNREVIEWED

                entry = self.labels.get(stem, {})
                existing = entry.get("human_label")
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
                    cell["label"].configure(image="", text="missing")
            else:
                cell["stem"] = None
                cell["frame"].configure(highlightbackground=COLOR_BG)
                cell["label"].configure(image="", bg=COLOR_BG)

    def _on_select(self, cell_idx):
        self.selected_cell = cell_idx
        for i, cell in enumerate(self.cells):
            if i == cell_idx and cell["stem"]:
                cell["frame"].configure(highlightbackground=COLOR_SELECTED)

    def _batch_label(self, label):
        cid = self.cluster_ids[self.cluster_idx]
        stems = self.cluster_to_images[cid]
        start = self.page_offset * CELLS_PER_PAGE
        page_stems = stems[start : start + CELLS_PER_PAGE]
        for stem in page_stems:
            if stem in self.labels:
                self.labels[stem]["human_label"] = label
        save_json(self.labels_path, self.labels)
        self._render_page()
        print(f"  Page marked as '{label}' — {len(page_stems)} images")

    def _individual_label(self, label):
        if self.selected_cell is None:
            return
        cell = self.cells[self.selected_cell]
        stem = cell["stem"]
        if stem and stem in self.labels:
            self.labels[stem]["human_label"] = label
        save_json(self.labels_path, self.labels)
        print(f"  {stem}: marked '{label}'")

    def _next_page(self):
        cid = self.cluster_ids[self.cluster_idx]
        stems = self.cluster_to_images[cid]
        max_page = (len(stems) - 1) // CELLS_PER_PAGE
        if self.page_offset < max_page:
            self.page_offset += 1
        elif self.cluster_idx < len(self.cluster_ids) - 1:
            self._load_cluster(self.cluster_idx + 1)
            return
        self._render_page()

    def _prev_page(self):
        if self.page_offset > 0:
            self.page_offset -= 1
        elif self.cluster_idx > 0:
            self._load_cluster(self.cluster_idx - 1)
            return
        self._render_page()


def main():
    cfg = load_config()
    images_dir = cfg["paths"]["images_dir"]
    labels_path = cfg["review"]["combined_labels_file"]
    clusters_path = cfg["review"]["cluster_labels_file"]

    root_dir = Path(__file__).resolve().parent.parent
    if not os.path.isabs(labels_path):
        labels_path = str(root_dir / labels_path)
    if not os.path.isabs(clusters_path):
        clusters_path = str(root_dir / clusters_path)

    root = tk.Tk()
    root.geometry("900x700")
    ClusterReviewGUI(root, images_dir, labels_path, clusters_path)
    root.mainloop()


if __name__ == "__main__":
    main()
