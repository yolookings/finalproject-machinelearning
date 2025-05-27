import os
import runpy # Modul untuk menjalankan skrip Python lain

# --- Bagian Penting ---
# Atur variabel lingkungan SEBELUM main.py diimpor atau dijalankan.
# Ini memberi tahu Flask untuk tidak menggunakan mode debug.
os.environ['FLASK_DEBUG'] = '0'

# Pastikan aplikasi mendengarkan pada port yang benar untuk Hugging Face Spaces
# Jika main.py Anda sudah menghandle port dari os.environ.get("PORT"), ini mungkin tidak perlu
# dimodifikasi secara eksplisit di sini, tapi FLASK_DEBUG tetap penting.
# Anda juga bisa mengatur variabel PORT di sini jika main.py Anda tidak menanganinya,
# tetapi lebih baik main.py yang menanganinya.

print("🚀 Starting application via run_hf.py with FLASK_DEBUG=0...")

# Jalankan main.py seolah-olah itu adalah skrip utama
# Ini akan mengeksekusi kode di main.py dalam konteks saat ini,
# di mana FLASK_DEBUG sudah diatur ke '0'.
try:
    # Jika main.py Anda memiliki app.run() di dalam if __name__ == '__main__':
    runpy.run_path("main.py", run_name="__main__")
except Exception as e:
    print(f"Error running main.py: {e}")
    # Mungkin tambahkan logging error yang lebih detail atau exit
    raise