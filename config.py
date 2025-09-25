# Global config
DEVICE       = "cpu"       # default to CPU; set "cuda" if GPU available
POSE_MODEL   = "yolov8n-pose.pt"
DET_MODEL    = "yolov8n.pt"
FPS_TARGET   = 30

# Camera assignment per ROSA section (A: chair, B: monitor, C: peripherals)
CAMERA_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
}

# Mouse hand preference for Section C
SECTIONC_HAND = "right"

# Ekspor data riset
EXPORT_CSV   = "rosa_export.csv"
EXPORT_JSONL = "rosa_export.jsonl"
