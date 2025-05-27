# run_hf.py
import os
import sys

# Dapatkan nama file skrip ini untuk logging yang lebih jelas
script_name = os.path.basename(__file__)

# Tambahkan direktori skrip saat ini ke sys.path.
# Ini membantu Python menemukan 'main.py' untuk diimpor,
# terutama jika Hugging Face Spaces menjalankan skrip dari direktori kerja yang berbeda.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir) # penting agar 'from main import app' berhasil

print(f"🚀 [{script_name}] Memulai. Mencoba mengimpor 'app' dari 'main.py'.")

try:
    # --- Langkah 1: Impor objek 'app' Flask dari main.py ---
    # Ini akan mengeksekusi semua kode di level modul main.py:
    # - Semua import (misalnya, tensorflow, deepface).
    # - Definisi variabel global, fungsi, dan kelas.
    # - Pembuatan instance Flask: `app = Flask(__name__)`.
    # - Pembuatan direktori UPLOAD_FOLDER.
    # - Definisi rute: `@app.route(...)`.
    # - Panggilan fungsi inisialisasi (misalnya, DeepFace yang membuat direktori .deepface).
    # PENTING: Blok `if __name__ == '__main__':` di main.py TIDAK akan dieksekusi di sini.
    from main import app  # Ini akan mencari main.py di direktori yang sama dengan run_hf.py
    print(f"✔ [{script_name}] Berhasil mengimpor 'app' dari 'main.py'.")

    # --- Langkah 2: Konfigurasi port dan host untuk server Flask ---
    # Hugging Face Spaces biasanya mengatur variabel lingkungan PORT.
    # Gunakan port 7860 sebagai default jika PORT tidak diset (7860 adalah port umum untuk Gradio/HF).
    port = int(os.environ.get("PORT", 7860)) # HF akan menyediakan PORT, default ke 7860
    # Selalu gunakan '0.0.0.0' agar server dapat diakses dari luar container Space.
    host = '0.0.0.0'

    print(f"🌍 [{script_name}] Akan menjalankan server Flask untuk 'app' di {host}:{port}.")
    print(f"   Mode Debug akan: False, Reloader akan: False.")

    # --- Langkah 3: Jalankan server Flask dengan pengaturan yang benar ---
    # Panggil app.run() secara langsung dari sini.
    # - debug=False: Mematikan mode debug Flask.
    # - use_reloader=False: Secara eksplisit mematikan reloader file.
    #   Ini sangat penting untuk mencegah loop restart, bahkan jika debug=False.
    # - threaded=True: Sesuai dengan konfigurasi di main.py Anda.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

except ImportError as e:
    # Blok ini dijalankan jika `from main import app` gagal.
    print(f"❌ [{script_name}] GAGAL: Tidak dapat mengimpor 'app' dari 'main.py'. Error: {e}")
    print(f"   Pastikan 'main.py' mendefinisikan instance Flask global bernama 'app' dan berada di direktori yang sama dengan '{script_name}'.")
    print(f"   Periksa juga log untuk error impor lainnya dari 'main.py'.")
    sys.exit(1) # Keluar dengan kode error agar jelas di log HF.

except Exception as e:
    # Menangkap error tak terduga lainnya selama proses.
    print(f"❌ [{script_name}] Terjadi error tak terduga: {e}")
    import traceback
    traceback.print_exc() # Cetak traceback lengkap untuk debugging.
    sys.exit(1) # Keluar dengan kode error.