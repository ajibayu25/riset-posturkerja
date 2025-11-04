# Global config
DEVICE       = "cpu"       # default to CPU; set "cuda" if GPU available
POSE_MODEL   = "yolov8n-pose.pt"
DET_MODEL    = "yolov8n.pt"
EARPHONE_MODEL = "yolo-earphone.pt"
FPS_TARGET   = 30

# Camera assignment per ROSA section (A: chair, B: monitor, C: peripherals)
CAMERA_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
}

# Friendly camera names exposed in the GUI OptionMenus.
# Update indices so they match the OS device order you see in Device Manager.
# Leave the "None" entry in place so a panel can be disabled.
CAMERA_PRESETS = [
    ("None", None),
    ("HD Webcam", 0),
    ("Logi C270 HD WebCam", 1),
]

# Mouse hand preference for Section C
SECTIONC_HAND = "right"

# Ekspor data riset
EXPORT_CSV   = "rosa_export.csv"
EXPORT_JSONL = "rosa_export.jsonl"
EXPORT_XLSX  = "rosa_summary.xlsx"
