import numpy as np

"""Lookup grids digitised from the official ROSA scoring form.

Each grid converts the summed sub-scores for a section into the
corresponding ROSA area or final score. Axes follow the notation in the
paperwork: rows are typically the vertical axis values listed on the left
of the chart, columns the horizontal axis values along the top.
"""

# === Section A (Chair) ===
# Horizontal axis: Armrest + Back support (values 2..9)
# Vertical axis: Seat pan height + seat depth (values 2..8)
SECTION_A_HORIZONTAL_AXIS = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
SECTION_A_VERTICAL_AXIS = np.array([2, 3, 4, 5, 6, 7, 8], dtype=int)
SECTION_A_GRID = np.array([
    [2, 2, 3, 4, 5, 6, 7, 8],
    [2, 2, 3, 4, 5, 6, 7, 8],
    [3, 3, 3, 4, 5, 6, 7, 8],
    [4, 4, 4, 4, 5, 6, 7, 8],
    [5, 5, 5, 5, 6, 7, 8, 9],
    [6, 6, 6, 7, 7, 8, 8, 9],
    [7, 7, 7, 8, 8, 9, 9, 9],
], dtype=int)

# === Section B (Monitor & Telephone) ===
# Rows: telephone stance (0..6) ; Columns: monitor posture sum (0..7)
SECTION_B_PHONE_AXIS = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
SECTION_B_MONITOR_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
MONITOR_PHONE_GRID = np.array([
    [1, 1, 1, 2, 3, 4, 5, 6],
    [1, 1, 2, 2, 3, 4, 5, 6],
    [1, 2, 2, 3, 3, 4, 6, 7],
    [2, 2, 3, 3, 4, 5, 6, 8],
    [3, 3, 4, 4, 5, 6, 7, 8],
    [4, 4, 5, 5, 6, 7, 8, 9],
    [5, 5, 6, 7, 8, 8, 9, 9],
], dtype=int)

# === Section C (Mouse & Keyboard) ===
# Rows: mouse posture sum (0..7) ; Columns: keyboard posture sum (0..7)
SECTION_C_MOUSE_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
SECTION_C_KEYBOARD_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
SECTIONC_MOUSE_KEYBOARD_GRID = np.array([
    [1, 1, 1, 2, 3, 4, 5, 6],
    [1, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 2, 3, 4, 5, 6, 7],
    [2, 3, 3, 3, 5, 6, 7, 8],
    [3, 4, 4, 5, 5, 6, 7, 8],
    [4, 5, 5, 6, 6, 7, 8, 9],
    [5, 6, 6, 7, 7, 8, 8, 9],
    [6, 7, 7, 8, 8, 9, 9, 9],
], dtype=int)

# === Monitor & Peripherals combo (Section B vs C) ===
# Rows: Section B score (1..9) ; Columns: Section C score (1..9)
MONITOR_PERIPHERALS_AXIS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MONITOR_PERIPHERALS_GRID = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [2, 2, 3, 4, 5, 6, 7, 8, 9],
    [3, 3, 3, 4, 5, 6, 7, 8, 9],
    [4, 4, 4, 4, 5, 6, 7, 8, 9],
    [5, 5, 5, 5, 5, 6, 7, 8, 9],
    [6, 6, 6, 6, 6, 6, 7, 8, 9],
    [7, 7, 7, 7, 7, 7, 7, 8, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
], dtype=int)

# === Final ROSA score (Chair vs Peripherals & Monitor) ===
# Rows: Chair score (1..10) ; Columns: Monitor & Peripherals score (1..10)
ROSA_FINAL_AXIS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
ROSA_FINAL_GRID = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 3, 3, 4, 5, 6, 7, 8, 9, 10],
    [4, 4, 4, 4, 5, 6, 7, 8, 9, 10],
    [5, 5, 5, 5, 5, 6, 7, 8, 9, 10],
    [6, 6, 6, 6, 6, 6, 7, 8, 9, 10],
    [7, 7, 7, 7, 7, 7, 7, 8, 9, 10],
    [8, 8, 8, 8, 8, 8, 8, 8, 9, 10],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 10],
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
], dtype=int)
