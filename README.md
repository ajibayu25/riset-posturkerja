# Real-Time ROSA (Rapid Office Strain Assessment)

Aplikasi ini adalah alat bantu untuk melakukan penilaian postur kerja di lingkungan perkantoran secara *real-time* menggunakan metode *Rapid Office Strain Assessment* (ROSA). Sistem ini memanfaatkan kamera untuk menganalisis postur tubuh, posisi monitor, dan penggunaan perangkat input (keyboard, mouse) untuk menghitung skor ROSA secara otomatis.

## Fitur Utama

- **Penilaian Postur Real-Time**: Menganalisis postur leher, punggung, dan lengan secara langsung melalui kamera.
- **Analisis Multi-Sudut**: Menggunakan beberapa kamera untuk mendapatkan pandangan dari sisi (kursi dan postur tubuh), depan (monitor dan keyboard), dan atas (perangkat peripheral).
- **Deteksi Objek**: Menggunakan model YOLOv8 untuk mendeteksi posisi tangan, penggunaan earphone, dan objek relevan lainnya.
- **Antarmuka Grafis (GUI)**: Dilengkapi dengan GUI berbasis Tkinter untuk memudahkan pemantauan dan konfigurasi dari ketiga bagian (A, B, dan C) secara bersamaan.
- **Mode CLI**: Mendukung eksekusi melalui *command-line* untuk pengujian atau penggunaan pada satu bagian spesifik.
- **Ekspor Data**: Hasil penilaian dapat diekspor ke format CSV, JSONL, dan XLSX untuk analisis lebih lanjut.
- **Konfigurasi Fleksibel**: Pengaturan kamera, model, dan preferensi lainnya dapat dengan mudah diubah melalui file `config.py`.

## Instalasi

1.  **Clone Repository**:
    ```bash
    git clone <URL_REPOSITORY_ANDA>
    cd riset-posturkerja
    ```

2.  **Buat Virtual Environment** (Direkomendasikan):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate  # Windows
    ```

3.  **Install Dependensi**:
    Proyek ini membutuhkan beberapa library Python. Anda bisa menginstalnya dengan `pip`. Pastikan Anda melihat file `requirements.txt` (jika ada) atau menginstal library utama yang digunakan seperti:
    ```bash
    pip install opencv-python ultralytics pandas openpyxl
    ```
    *Catatan: Mungkin ada dependensi lain yang dibutuhkan. Silakan periksa import di dalam kode untuk daftar lengkap.*

## Konfigurasi

Sebelum menjalankan aplikasi, konfigurasikan beberapa parameter penting di dalam file `config.py`:

- **`DEVICE`**: Atur ke `"cuda"` jika Anda memiliki GPU yang mendukung, atau biarkan `"cpu"`.
- **`CAMERA_PRESETS`**: Sesuaikan nama dan indeks kamera yang terhubung ke komputer Anda. Indeks dimulai dari `0`.
- **`CAMERA_DEFAULTS`**: Tentukan kamera default untuk setiap bagian (A, B, C) berdasarkan nama yang ada di `CAMERA_PRESETS`.
- **`GLARE_SERIAL_PORT`**: Jika Anda menggunakan sensor silau eksternal (Arduino), atur port serial yang sesuai (misal: `"COM5"`).
- **`EXPORT_...`**: Tentukan nama file untuk ekspor data.

## Penggunaan

Aplikasi ini dapat dijalankan dalam dua mode:

### 1. Mode GUI (Multi-Section)

Mode ini akan membuka jendela aplikasi yang menampilkan feed dari tiga kamera (jika dikonfigurasi) untuk Bagian A, B, dan C secara bersamaan.

Untuk menjalankan mode GUI, eksekusi perintah berikut di terminal:
```bash
python main.py
```
atau
```bash
python main.py --mode multi
```

### 2. Mode CLI (Single-Section)

Mode ini berguna untuk fokus pada satu bagian penilaian saja (misalnya, hanya postur kursi).

Gunakan argumen `--mode single` dan tentukan bagian serta kamera yang ingin digunakan:
```bash
python main.py --mode single --section <a|b|c> --cam <indeks_kamera>
```
**Contoh**:
- Menjalankan penilaian Bagian A (kursi) menggunakan kamera dengan indeks `0`:
  ```bash
  python main.py --mode single --section a --cam 0
  ```
- Menjalankan penilaian Bagian B (monitor) menggunakan kamera dengan indeks `2`:
  ```bash
  python main.py --mode single --section b --cam 2
  ```

## Struktur Folder

- **`/assets`**: Berisi gambar referensi yang digunakan dalam aplikasi.
- **`/constants`**: Menyimpan konstanta seperti grid dan ambang batas sudut.
- **`/core`**: Modul inti untuk geometri, penghalusan data, dan timer.
- **`/gui`**: Kode untuk antarmuka pengguna (GUI) berbasis Tkinter.
- **`/models`**: Skrip untuk memuat dan menjalankan model deteksi (pose, tangan, dll.).
- **`/rosa_io`**: Modul untuk menangani ekspor data ke berbagai format.
- **`/scoring`**: Logika utama untuk menghitung skor ROSA untuk setiap bagian (A, B, C) dan total.
- **`/sensory`**: Kode untuk berinteraksi dengan sensor eksternal seperti sensor silau.
- **`/snapshots`**: Folder default untuk menyimpan gambar hasil tangkapan.
- **`config.py`**: File konfigurasi utama.
- **`main.py`**: Titik masuk utama aplikasi.
- **`*.pt`**: File bobot model machine learning (YOLO).