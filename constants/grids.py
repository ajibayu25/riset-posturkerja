import numpy as np  # Import library NumPy untuk membuat array numerik dan matriks (grid lookup)

"""
Lookup grids digitised from the official ROSA scoring form.

Setiap grid di bawah adalah versi digital dari tabel konversi skor ROSA resmi.
Masing-masing grid mengonversi jumlah sub-skor dari suatu bagian menjadi skor area ROSA.
Arah sumbu mengikuti formulir ROSA:
- Baris (rows) = nilai vertikal (biasanya di sisi kiri tabel)
- Kolom (columns) = nilai horizontal (biasanya di atas tabel)
"""

# === Section A (Chair) ===
# Menilai postur kursi: tinggi kursi, kedalaman dudukan, sandaran, dan sandaran tangan.

# Sumbu horizontal = jumlah skor Armrest + Back support (2 sampai 9)
SECTION_A_HORIZONTAL_AXIS = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

# Sumbu vertikal = jumlah skor Seat pan height + Seat pan depth (2 sampai 8)
SECTION_A_VERTICAL_AXIS = np.array([2, 3, 4, 5, 6, 7, 8], dtype=int)

# Grid konversi (7x8): hasil skor Section A untuk kombinasi nilai sumbu vertikal × horizontal
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
# Menilai postur leher dan posisi kepala saat menggunakan monitor dan telepon.

# Sumbu vertikal (rows) = telephone stance (0–6)
SECTION_B_PHONE_AXIS = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)

# Sumbu horizontal (columns) = monitor posture sum (0–7)
SECTION_B_MONITOR_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)

# Grid konversi (7x8): hasil skor Section B untuk kombinasi nilai telepon dan monitor
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
# Menilai postur tangan dan pergelangan tangan saat menggunakan mouse dan keyboard.

# Sumbu vertikal (rows) = mouse posture sum (0–7)
SECTION_C_MOUSE_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)

# Sumbu horizontal (columns) = keyboard posture sum (0–7)
SECTION_C_KEYBOARD_AXIS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)

# Grid konversi (8x8): hasil skor Section C untuk kombinasi postur mouse dan keyboard
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
# Menggabungkan skor Section B dan Section C menjadi "Monitor & Peripherals score".

# Sumbu umum (1–9) untuk keduanya
MONITOR_PERIPHERALS_AXIS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

# Grid konversi (9x9): hasil kombinasi skor Section B dan Section C
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
# Menggabungkan skor Section A (Chair) dan skor gabungan B+C (Monitor & Peripherals)
# menjadi skor akhir ROSA (1–10).

# Sumbu skor akhir 1–10
ROSA_FINAL_AXIS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)

# Grid konversi (10x10): hasil akhir skor ROSA total
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
    [10,10,10,10,10,10,10,10,10,10],
], dtype=int)
