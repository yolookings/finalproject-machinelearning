from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
import uuid
from PIL import Image, UnidentifiedImageError # Import Pillow and specific error
import pillow_heif

pillow_heif.register_heif_opener()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic', 'heif'}
MAX_IMAGE_SIZE_MB = 10  # Max 10 MB per image

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE_MB * 1024 * 1024 * 2 # For two images + overhead

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_heic_to_jpg(heic_path, upload_folder):
    """Konversi file HEIC ke JPG dan kembalikan path JPG baru."""
    try:
        img = Image.open(heic_path) # Pillow opens HEIC due to pillow_heif
        
        # Create a unique JPG filename in the same upload folder
        jpg_filename = os.path.splitext(os.path.basename(heic_path))[0] + "_" + str(uuid.uuid4())[:4] + ".jpg"
        jpg_path = os.path.join(upload_folder, jpg_filename)
        
        img.convert('RGB').save(jpg_path, "JPEG")
        app.logger.info(f"Converted {heic_path} to {jpg_path}")
        return jpg_path
    except UnidentifiedImageError:
        app.logger.error(f"Cannot identify image file (not a valid HEIC/image): {heic_path}")
        return None
    except Exception as e:
        app.logger.error(f"Error converting HEIC {heic_path}: {e}")
        return None

@app.route('/')
def index():
    # Pass template name from function argument
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Harap unggah kedua gambar'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Nama file tidak boleh kosong'}), 400

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': f'Jenis file tidak diizinkan. Hanya {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Note: MAX_CONTENT_LENGTH already checked by Flask for total request size.
    # Individual file size check can be done by reading file.tell() after saving or from file.content_length
    # For simplicity, we'll rely on MAX_CONTENT_LENGTH for now.

    # Save files temporarily with unique names
    # Use original extension for saving, then check for HEIC
    ext1 = os.path.splitext(file1.filename)[1].lower()
    ext2 = os.path.splitext(file2.filename)[1].lower()
    
    original_filename1 = str(uuid.uuid4()) + ext1
    original_filename2 = str(uuid.uuid4()) + ext2

    img1_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename1)
    img2_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename2)

    files_to_delete = []

    try:
        file1.save(img1_path_original)
        files_to_delete.append(img1_path_original)
        file2.save(img2_path_original)
        files_to_delete.append(img2_path_original)

        img1_path_for_deepface = img1_path_original
        img2_path_for_deepface = img2_path_original
        
        # Check and convert if HEIC
        if ext1 in ('.heic', '.heif'):
            app.logger.info(f"Image 1 ({original_filename1}) is HEIC, attempting conversion.")
            converted_path1 = convert_heic_to_jpg(img1_path_original, app.config['UPLOAD_FOLDER'])
            if converted_path1:
                img1_path_for_deepface = converted_path1
                files_to_delete.append(converted_path1)
            else:
                return jsonify({'error': 'Gagal mengkonversi Gambar 1 (HEIC/HEIF). Pastikan file valid.'}), 500
        
        if ext2 in ('.heic', '.heif'):
            app.logger.info(f"Image 2 ({original_filename2}) is HEIC, attempting conversion.")
            converted_path2 = convert_heic_to_jpg(img2_path_original, app.config['UPLOAD_FOLDER'])
            if converted_path2:
                img2_path_for_deepface = converted_path2
                files_to_delete.append(converted_path2)
            else:
                return jsonify({'error': 'Gagal mengkonversi Gambar 2 (HEIC/HEIF). Pastikan file valid.'}), 500

        # Verify using the (potentially converted) image paths
        result = DeepFace.verify(
            img1_path=img1_path_for_deepface,
            img2_path=img2_path_for_deepface,
            model_name="VGG-Face", # Or make this configurable
            detector_backend='retinaface', # Often a good default
            enforce_detection=False # Important: allows processing even if one face is hard to detect initially
                                    # Rely on subsequent error for "no face detected"
        )
        
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'similarity_metric': result['similarity_metric']
        })

    except ValueError as ve: 
        app.logger.error(f"ValueError during DeepFace processing: {ve}")
        if "Face could not be detected" in str(ve) or "cannot be aligned" in str(ve) or "No face detected in" in str(ve):
             return jsonify({'error': f'Wajah tidak dapat terdeteksi pada salah satu atau kedua gambar. Detail: {str(ve)}'}), 400 # 400 as it's a client-side image issue
        return jsonify({'error': f'Terjadi kesalahan pemrosesan gambar: {str(ve)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error during DeepFace processing: {e}", exc_info=True)
        return jsonify({'error': f'Terjadi kesalahan internal server: {str(e)}'}), 500
    finally:
        # Clean up all temporary files
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    app.logger.info(f"Successfully deleted temp file: {f_path}")
                except Exception as e_del:
                    app.logger.error(f"Error deleting temp file {f_path}: {e_del}")

if __name__ == '__main__':
    # For development, ensure Flask's logger is set up if not using default print
    import logging
    logging.basicConfig(level=logging.INFO) # Show INFO level logs from app.logger
    # When using a production server like Gunicorn, it handles logging.
    app.run(debug=True, host='0.0.0.0', port=5000)