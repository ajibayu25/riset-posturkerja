# Global config
DEVICE       = "cpu"       # default to CPU; set "cuda" if GPU available
POSE_MODEL   = "yolov8n-pose.pt"
DET_MODEL    = "yolov8n.pt"
EARPHONE_MODEL = "yolo-earphone.pt"
HAND_MODEL  = "yolo-handdetection.pt"
FPS_TARGET   = 30

# Camera assignment per ROSA section (A: chair, B: monitor, C: peripherals)
CAMERA_INDEX = {
    "A": 0,  # Side view camera index (fallback if name matching fails)
    "B": 2,  # Front view fallback index
    "C": 1,  # Overhead view fallback index
}

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

# Ekspor data riset
EXPORT_CSV   = "rosa_export.csv"
EXPORT_JSONL = "rosa_export.jsonl"
EXPORT_XLSX  = "rosa_summary.xlsx"
