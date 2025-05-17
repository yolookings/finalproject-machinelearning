from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
import uuid # Untuk nama file unik
from PIL import Image # Import Pillow
import pillow_heif # Untuk membaca HEIC

# Daftarkan HEIF opener dengan Pillow agar Image.open() bisa mengenali HEIC
pillow_heif.register_heif_opener()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def convert_heic_to_jpg(heic_path):
    """Konversi file HEIC ke JPG dan kembalikan path JPG baru."""
    try:
        # Buka file HEIC menggunakan Pillow (setelah HEIF opener diregistrasi)
        img = Image.open(heic_path)
        
        # Buat nama file JPG baru
        jpg_filename = os.path.splitext(os.path.basename(heic_path))[0] + "_" + str(uuid.uuid4())[:4] + ".jpg"
        jpg_path = os.path.join(os.path.dirname(heic_path), jpg_filename)
        
        # Konversi ke RGB (penting untuk format JPG) dan simpan
        img.convert('RGB').save(jpg_path, "JPEG")
        print(f"Converted {heic_path} to {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"Error converting HEIC {heic_path}: {e}")
        return None # Kembalikan None jika konversi gagal

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Harap unggah kedua gambar'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Nama file tidak boleh kosong'}), 400

    # Simpan file sementara dengan nama unik
    original_filename1 = str(uuid.uuid4()) + os.path.splitext(file1.filename)[1]
    original_filename2 = str(uuid.uuid4()) + os.path.splitext(file2.filename)[1]

    img1_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename1)
    img2_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename2)

    file1.save(img1_path_original)
    file2.save(img2_path_original)

    img1_path_for_deepface = img1_path_original
    img2_path_for_deepface = img2_path_original
    
    # Daftar file yang perlu dihapus di akhir
    files_to_delete = [img1_path_original, img2_path_original]

    # Cek dan konversi jika HEIC
    if original_filename1.lower().endswith(('.heic', '.heif')):
        converted_path1 = convert_heic_to_jpg(img1_path_original)
        if converted_path1:
            img1_path_for_deepface = converted_path1
            files_to_delete.append(converted_path1) # Tambahkan file konversi untuk dihapus
        else:
            # Hapus file asli jika konversi gagal
            for f_path in files_to_delete:
                if os.path.exists(f_path): os.remove(f_path)
            return jsonify({'error': f'Gagal mengkonversi gambar 1 (HEIC).'}), 500
    
    if original_filename2.lower().endswith(('.heic', '.heif')):
        converted_path2 = convert_heic_to_jpg(img2_path_original)
        if converted_path2:
            img2_path_for_deepface = converted_path2
            files_to_delete.append(converted_path2) # Tambahkan file konversi untuk dihapus
        else:
            # Hapus file asli dan konversi (jika ada) jika konversi gagal
            for f_path in files_to_delete:
                if os.path.exists(f_path): os.remove(f_path)
            return jsonify({'error': f'Gagal mengkonversi gambar 2 (HEIC).'}), 500

    try:
        result = DeepFace.verify(img1_path=img1_path_for_deepface,
                                 img2_path=img2_path_for_deepface,
                                 model_name="VGG-Face",
                                 enforce_detection=False)
        
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'similarity_metric': result['similarity_metric']
        })

    except Exception as e:
        print(f"Error during DeepFace processing: {e}")
        if "Face could not be detected" in str(e) or "cannot be aligned" in str(e):
             return jsonify({'error': f'Wajah tidak dapat terdeteksi pada salah satu atau kedua gambar. Detail: {str(e)}'}), 500
        return jsonify({'error': f'Terjadi kesalahan saat pemrosesan: {str(e)}'}), 500
    finally:
        # Hapus semua file (asli dan yang dikonversi jika ada)
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"Successfully deleted {f_path}")
                except Exception as e_del:
                    print(f"Error deleting file {f_path}: {e_del}")


if __name__ == '__main__':
    app.run(debug=True)