# Real-Time ROSA (Rapid Office Strain Assessment)

This application is an auxiliary tool for conducting real-time posture assessment in an office environment using the Rapid Office Strain Assessment (ROSA) method. The system utilizes cameras to analyze body posture, monitor position, and the use of input devices (keyboard, mouse) to automatically calculate the ROSA score.

## Key Features

- **Real-Time Posture Assessment**: Analyzes neck, back, and arm posture directly through a camera.
- **Multi-Angle Analysis**: Uses multiple cameras to get views from the side (chair and body posture), front (monitor and keyboard), and top (peripheral devices).
- **Object Detection**: Employs the YOLOv8 model to detect hand positions, earphone usage, and other relevant objects.
- **Graphical User Interface (GUI)**: Equipped with a Tkinter-based GUI for easy monitoring and configuration of all three sections (A, B, and C) simultaneously.
- **CLI Mode**: Supports execution via the command-line for testing or running a single, specific section.
- **Data Export**: Assessment results can be exported to CSV, JSONL, and XLSX formats for further analysis.
- **Flexible Configuration**: Camera settings, models, and other preferences can be easily modified in the `config.py` file.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd riset-posturkerja
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies**:
    This project requires several Python libraries. You can install them using `pip`. Make sure to check for a `requirements.txt` file or install the main libraries used, such as:
    ```bash
    pip install opencv-python ultralytics pandas openpyxl
    ```
    *Note: Other dependencies might be required. Please check the imports in the code for a complete list.*

## Configuration

Before running the application, configure several important parameters in the `config.py` file:

- **`DEVICE`**: Set to `"cuda"` if you have a supported GPU, or leave it as `"cpu"`.
- **`CAMERA_PRESETS`**: Adjust the names and indices of the cameras connected to your computer. The index starts from `0`.
- **`CAMERA_DEFAULTS`**: Specify the default camera for each section (A, B, C) based on the names in `CAMERA_PRESETS`.
- **`GLARE_SERIAL_PORT`**: If you are using an external glare sensor (Arduino), set the appropriate serial port (e.g., `"COM5"`).
- **`EXPORT_...`**: Define the filenames for data export.

## Usage

The application can be run in two modes:

### 1. GUI Mode (Multi-Section)

This mode opens an application window displaying feeds from three cameras (if configured) for Sections A, B, and C simultaneously.

To run in GUI mode, execute the following command in the terminal:
```bash
python main.py
```
or
```bash
python main.py --mode multi
```

### 2. CLI Mode (Single-Section)

This mode is useful for focusing on a single assessment section (e.g., only the chair posture).

Use the `--mode single` argument and specify the section and camera you want to use:
```bash
python main.py --mode single --section <a|b|c> --cam <camera_index>
```
**Examples**:
- Run Section A assessment (chair) using the camera with index `0`:
  ```bash
  python main.py --mode single --section a --cam 0
  ```
- Run Section B assessment (monitor) using the camera with index `2`:
  ```bash
  python main.py --mode single --section b --cam 2
  ```

## Folder Structure

- **`/assets`**: Contains reference images used in the application.
- **`/constants`**: Stores constants like grids and angle thresholds.
- **`/core`**: Core modules for geometry, data smoothing, and timers.
- **`/gui`**: Code for the Tkinter-based graphical user interface (GUI).
- **`/models`**: Scripts for loading and running detection models (pose, hands, etc.).
- **`/rosa_io`**: Modules for handling data export to various formats.
- **`/scoring`**: The main logic for calculating ROSA scores for each section (A, B, C) and the total.
- **`/sensory`**: Code for interacting with external sensors like the glare sensor.
- **`/snapshots`**: Default folder for saving captured images.
- **`config.py`**: The main configuration file.
- **`main.py`**: The main entry point of the application.
- **`*.pt`**: Machine learning model weight files (YOLO).
