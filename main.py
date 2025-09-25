"""
Main entry point for ROSA real-time.
Supports multi-section GUI (default) or single-section CLI.
"""
import argparse
import sys
import tkinter as tk

from config import CAMERA_INDEX, SECTIONC_HAND
from gui.app_tk import MultiSectionTkApp
from scoring.sectiona import LiveSectionAApp
from scoring.sectionb import LiveSectionBApp
from scoring.sectionc import LiveSectionCApp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["multi", "single"], default="multi", help="jalankan GUI multi-section atau mode single")
    p.add_argument("--section", choices=["a", "b", "c"], help="pilih section untuk mode single")
    p.add_argument("--cam", type=int, default=0, help="index kamera untuk mode single")
    p.add_argument("--export", choices=["csv", "json", "none"], default="csv")
    return p.parse_args()


def run_multi() -> None:
    root = tk.Tk()
    MultiSectionTkApp(root)
    root.mainloop()


def run_single(section: str, cam_index: int, export: str) -> None:
    section = section.lower()
    if section == "a":
        app = LiveSectionAApp(cam_index=cam_index, export_mode=export)
    elif section == "b":
        app = LiveSectionBApp(cam_index=cam_index, export_mode=export)
    else:
        app = LiveSectionCApp(cam_index=cam_index, export_mode=export)
    app.run()


def main():
    args = parse_args()
    if args.mode == "multi":
        run_multi()
        return

    section = args.section
    if section is None:
        section = input("Pilih Section (A/B/C): ").strip().lower()
        if section not in {"a", "b", "c"}:
            print("Section tidak valid.", file=sys.stderr)
            return
    run_single(section, args.cam, args.export)


if __name__ == "__main__":
    main()
