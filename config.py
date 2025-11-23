# Global config
from datetime import datetime
from pathlib import Path

DEVICE       = "cpu"       # default to CPU; set "cuda" if GPU available
POSE_MODEL   = "yolov8n-pose.pt"
DET_MODEL    = "yolov8n.pt"
EARPHONE_MODEL = "yolo-earphone.pt"
HAND_MODEL  = "yolo-handdetection.pt"

# Camera assignment per ROSA section (A: chair, B: monitor, C: peripherals)
CAMERA_INDEX = {
    "A": 0,  # Side view camera index (fallback if name matching fails)
    "B": 2,  # Front view fallback index
    "C": 1,  # Overhead view fallback index
}

# Default camera capture settings (kept low to reduce CPU/GPU load)
CAMERA_TARGET_FPS = 8
CAMERA_FRAME_WIDTH = 480
CAMERA_FRAME_HEIGHT = 270

# Unified sampling/export interval for ROSA scores (seconds)
DATA_CAPTURE_INTERVAL = 10.0

# Friendly camera names exposed in the GUI OptionMenus.
# Update indices so they match the OS device order you see in Device Manager.
# Leave the "None" entry in place so a panel can be disabled.
CAMERA_PRESETS = [
    ("None", None),
    ("HD Webcam", 0),
    ("Integrated Camera", 2),
    ("HD Pro Webcam C920", 1),
]

# Section-specific camera name preferences to keep defaults flexible.
# Each list entry is matched (case-insensitive substring) against the available
# camera labels reported by the OS. The first match is chosen automatically.
# Update the strings to reflect the devices you typically connect.
CAMERA_DEFAULTS = {
    "A": ["None"],
    "B": ["None"],
    "C": ["None"],
}

# Mouse hand preference for Section C
SECTIONC_HAND = "right"

# External glare detector (Arduino) serial settings
GLARE_SERIAL_PORT = None  # e.g. "COM5" or "/dev/ttyUSB0"
GLARE_BAUDRATE = 115200

# Ekspor data riset (per sesi, di bawah folder "rosa-exports")
# Pastikan path absolut supaya tidak tergantung working directory.
BASE_DIR = Path(__file__).resolve().parent
EXPORT_ROOT = BASE_DIR / "rosa-exports"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPORT_DIR = EXPORT_ROOT / SESSION_ID
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_CSV   = EXPORT_DIR / "rosa_export.csv"
EXPORT_JSONL = EXPORT_DIR / "rosa_export.jsonl"
EXPORT_XLSX  = EXPORT_DIR / "rosa_summary.xlsx"
