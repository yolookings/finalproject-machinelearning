from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import uuid
from PIL import Image, UnidentifiedImageError # Ditambahkan UnidentifiedImageError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Format log ditambahkan
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# Supported models and their descriptions
SUPPORTED_MODELS = {
    'VGG-Face': 'Fast and reliable, good for general use',
    'Facenet': 'Google\'s model, high accuracy',
    'Facenet512': 'Enhanced version of FaceNet with better performance',
    'OpenFace': 'Lightweight model, faster processing',
    'DeepFace': 'Facebook\'s model, good accuracy',
    'DeepID': 'Academic research model',
    'ArcFace': 'State-of-the-art model with excellent accuracy',
    'Dlib': 'Traditional computer vision approach',
    'SFace': 'Lightweight and fast model'
}

# Distance metrics for different models
# DeepFace biasanya memilih metrik default yang sesuai,
# tapi mendefinisikannya secara eksplisit bisa memberikan kontrol lebih atau untuk dokumentasi.
DISTANCE_METRICS = {
    'VGG-Face': 'cosine',
    'Facenet': 'cosine',
    'Facenet512': 'cosine',
    'OpenFace': 'cosine', # Sebenarnya bisa juga 'euclidean_l2'
    'DeepFace': 'cosine',
    'DeepID': 'cosine',
    'ArcFace': 'cosine', # Sebenarnya bisa juga 'euclidean_l2'
    'Dlib': 'euclidean',
    'SFace': 'cosine'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image(image_path, max_size=(800, 800), quality=85):
    """Optimize image for faster processing. Converts to RGB and saves as JPEG."""
    try:
        with Image.open(image_path) as img:
            logger.info(f"Optimizing image: {image_path}. Original mode: {img.mode}, size: {img.size}")
            # Convert to RGB if necessary (DeepFace typically expects RGB)
            if img.mode in ('RGBA', 'P', 'LA'): # LA adalah Luminance Alpha
                img = img.convert('RGB')
            elif img.mode != 'RGB': # Jika ada mode lain yang tidak umum
                logger.warning(f"Image {image_path} has an unusual mode {img.mode}. Attempting RGB conversion.")
                img = img.convert('RGB')

            # Resize if too large
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save optimized image (Perbaikan pada penamaan file)
            base, ext = os.path.splitext(image_path)
            # Selalu simpan sebagai .jpg karena kita konversi ke RGB dan DeepFace umumnya bekerja baik dengan JPG
            optimized_path = f"{base}_optimized.jpg"
            img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
            logger.info(f"Saved optimized image to: {optimized_path} with size {img.size}")
            return optimized_path
    except UnidentifiedImageError: # Error spesifik jika file bukan gambar
        logger.error(f"Cannot identify image file {image_path}. It might be corrupted or not an image.")
        return image_path # Kembalikan path asli jika gagal
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        return image_path # Kembalikan path asli jika gagal

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models_api(): # Dirubah nama fungsi agar tidak bentrok dengan variabel 'models'
    """Get list of supported models"""
    return jsonify({
        'models': SUPPORTED_MODELS,
        'default': 'VGG-Face' # Model default yang disarankan
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    files_to_delete = []
    request_id = str(uuid.uuid4()) # ID unik untuk setiap request, berguna untuk logging
    logger.info(f"[RequestID: {request_id}] Received prediction request.")

    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            logger.warning(f"[RequestID: {request_id}] Both images not found in request.")
            return jsonify({'error': 'Harap unggah kedua gambar'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']
        model_name = request.form.get('model', 'VGG-Face') # Default ke VGG-Face jika tidak disediakan

        logger.info(f"[RequestID: {request_id}] Files received: {file1.filename}, {file2.filename}. Model: {model_name}")

        if file1.filename == '' or file2.filename == '':
            logger.warning(f"[RequestID: {request_id}] One or both filenames are empty.")
            return jsonify({'error': 'Nama file tidak boleh kosong'}), 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            logger.warning(f"[RequestID: {request_id}] Unsupported file format detected.")
            return jsonify({'error': 'Format file tidak didukung. Harap gunakan JPG, PNG, GIF, BMP, atau WebP'}), 400

        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"[RequestID: {request_id}] Unsupported model selected: {model_name}")
            return jsonify({'error': f'Model tidak didukung. Model yang tersedia: {list(SUPPORTED_MODELS.keys())}'}), 400

        ext1 = os.path.splitext(file1.filename)[1].lower() if '.' in file1.filename else '.jpg'
        ext2 = os.path.splitext(file2.filename)[1].lower() if '.' in file2.filename else '.jpg'

        # Beri ekstensi default jika tidak ada, meskipun validasi 'allowed_file' seharusnya sudah menangkap ini
        filename1 = f"{request_id}_img1{ext1 if ext1 else '.tmp'}"
        filename2 = f"{request_id}_img2{ext2 if ext2 else '.tmp'}"

        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(img1_path)
        file2.save(img2_path)
        files_to_delete.extend([img1_path, img2_path])
        logger.info(f"[RequestID: {request_id}] Files saved: {img1_path}, {img2_path}")

        # Optimasi gambar (opsional tapi direkomendasikan)
        # Jika optimasi gagal, gunakan gambar asli
        optimized_img1 = optimize_image(img1_path)
        optimized_img2 = optimize_image(img2_path)

        if optimized_img1 != img1_path:
            files_to_delete.append(optimized_img1)
        if optimized_img2 != img2_path:
            files_to_delete.append(optimized_img2)

        process_img1 = optimized_img1
        process_img2 = optimized_img2

        distance_metric = DISTANCE_METRICS.get(model_name, 'cosine') # Default ke cosine
        logger.info(f"[RequestID: {request_id}] Using model: {model_name}, distance metric: {distance_metric} for verification.")
        logger.info(f"[RequestID: {request_id}] Image paths for DeepFace: {process_img1}, {process_img2}")


        # Parameter enforce_detection:
        # True: Hanya akan memproses jika wajah terdeteksi dengan pasti. Bisa lebih akurat, tapi lebih sering gagal jika wajah kurang jelas.
        # False: Akan mencoba memproses bahkan jika deteksi wajah kurang pasti. Lebih toleran, tapi bisa menghasilkan false positive/negative.
        # Untuk aplikasi pengguna akhir, False seringkali lebih baik dengan catatan kualitas gambar harus baik.
        enforce_detection_flag = False
        # detector_backend_choice = 'opencv' # Pilihan umum: 'opencv', 'ssd', 'mtcnn', 'dlib', 'retinaface'

        try:
            result = DeepFace.verify(
                img1_path=process_img1,
                img2_path=process_img2,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=enforce_detection_flag,
                # detector_backend=detector_backend_choice
            )
            logger.info(f"[RequestID: {request_id}] DeepFace verification result for {model_name}: Distance: {result['distance']:.4f}, Threshold: {result['threshold']:.4f}, Verified: {result['verified']}")

            # Kalkulasi similarity percentage (0-100)
            similarity_percentage = 0.0
            if result.get('threshold', 0) > 0: # Pastikan threshold tidak nol untuk menghindari ZeroDivisionError
                similarity_percentage = (1 - (result.get('distance', float('inf')) / result['threshold'])) * 100
            elif result.get('distance', float('inf')) == 0 : # Jika threshold 0, dan distance 0, maka 100% mirip
                 similarity_percentage = 100.0

            # Batasi similarity_percentage antara 0 dan 100
            similarity_percentage = max(0.0, min(100.0, similarity_percentage))

            response_data = {
                'verified': result['verified'],
                'distance': float(result['distance']),
                'threshold': float(result['threshold']),
                'model': result['model'], # Nama model yang digunakan oleh DeepFace (bisa berbeda jika ada fallback)
                'similarity_metric_used': result['similarity_metric'], # Metrik yang benar-benar digunakan
                'similarity_percentage': round(similarity_percentage, 2), # Skor kemiripan 0-100
                'facial_areas': { # Menambahkan informasi area wajah yang terdeteksi
                    'img1': result.get('facial_areas', {}).get('img1'),
                    'img2': result.get('facial_areas', {}).get('img2')
                },
                'model_description': SUPPORTED_MODELS.get(model_name, 'Model tidak dikenal'),
                'processing_details': {
                    'requested_model': model_name,
                    'requested_distance_metric': distance_metric,
                    'enforce_detection': enforce_detection_flag,
                    'detector_backend_used': result.get('detector_backend', 'default')
                }
            }
            return jsonify(response_data)

        except Exception as deepface_error:
            error_msg_lower = str(deepface_error).lower()
            logger.error(f"[RequestID: {request_id}] DeepFace processing error with {model_name} on images {process_img1}, {process_img2}: {deepface_error}", exc_info=True)

            user_error_message = f'Proses pengenalan wajah gagal: {str(deepface_error)}'
            user_suggestion = f'Coba gunakan model lain atau gambar yang berbeda. Model saat ini: {model_name}'

            if 'face could not be detected' in error_msg_lower or "couldn't find any face" in error_msg_lower:
                user_error_message = 'Wajah tidak dapat terdeteksi pada satu atau kedua gambar.'
                user_suggestion = 'Pastikan wajah terlihat jelas, menghadap depan, dan mendapat pencahayaan yang baik. Coba model seperti "OpenFace" atau "Dlib" yang mungkin lebih toleran.'
                status_code = 400
            elif 'dimension' in error_msg_lower or 'shape' in error_msg_lower or 'corrupt' in error_msg_lower:
                user_error_message = 'Kesalahan pemrosesan gambar. Gambar mungkin rusak atau dalam format yang tidak sepenuhnya didukung.'
                user_suggestion = 'Coba unggah gambar yang berbeda atau konversi ke format JPG/PNG standar.'
                status_code = 400
            elif 'memory' in error_msg_lower or 'out of memory' in error_msg_lower:
                user_error_message = 'Pemrosesan gagal karena batasan memori. Coba gunakan gambar dengan ukuran lebih kecil.'
                user_suggestion = 'Ubah ukuran gambar Anda menjadi lebih kecil (misalnya di bawah 1MB) dan coba lagi.'
                status_code = 500
            else: # Kesalahan DeepFace umum lainnya
                status_code = 500

            return jsonify({
                'error': user_error_message,
                'suggestion': user_suggestion,
                'details': str(deepface_error) # Untuk debugging di sisi developer jika diperlukan
            }), status_code

    except Exception as general_error:
        logger.error(f"[RequestID: {request_id}] General error in predict endpoint: {general_error}", exc_info=True)
        return jsonify({
            'error': 'Terjadi kesalahan tak terduga selama pemrosesan.',
            'details': str(general_error)
        }), 500

    finally:
        logger.info(f"[RequestID: {request_id}] Cleaning up temporary files: {files_to_delete}")
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"[RequestID: {request_id}] Deleted temporary file: {file_path}")
                except Exception as delete_error:
                    logger.warning(f"[RequestID: {request_id}] Could not delete file {file_path}: {delete_error}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Face Similarity API is running.',
        'supported_models_count': len(SUPPORTED_MODELS),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.errorhandler(413)
def too_large(e):
    logger.warning(f"File too large error (413): {e.description}")
    return jsonify({'error': 'File terlalu besar. Ukuran maksimum adalah 16MB per gambar.'}), 413

@app.errorhandler(404)
def not_found_error(e): # Dirubah nama fungsi agar tidak bentrok
    logger.info(f"Not found error (404) for path: {request.path}")
    return jsonify({'error': 'Endpoint tidak ditemukan'}), 404

@app.errorhandler(500)
def internal_server_error(e): # Dirubah nama fungsi agar tidak bentrok
    # Jika e adalah InternalServerError dari Werkzeug, e.original_exception mungkin memiliki info lebih
    original_exception = getattr(e, 'original_exception', None)
    logger.error(f"Internal server error (500): {e}. Original Exception: {original_exception}", exc_info=True)
    return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Memulai Server Analisis Kemiripan Wajah...")
    print(f"🕒 Waktu Startup: {logging.Formatter().formatTime(logging.LogRecord(None,None,'',0,'',(),None,None), datefmt='%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Folder Unggahan: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🤖 Model yang Didukung: {', '.join(SUPPORTED_MODELS.keys())}")
    print(f"📏 Ukuran File Maks: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"🛠️ Mode Debug Flask: {'Aktif' if app.debug else 'Nonaktif'}")
    print(f"🧵 Threading Flask: {'Aktif' if app.config.get('THREADED', False) else 'Nonaktif'}") # Cek config Flask untuk threaded
    print("🌐 Server akan tersedia di: http://localhost:5000 (atau alamat IP Anda jika host='0.0.0.0')")
    print("📝 Log akan dicatat dengan format: WAKTU - NAMA_MODUL - LEVEL - PESAN")
    print("="*60 + "\n")

    app.run(
        debug=True, # Nonaktifkan untuk produksi
        host='0.0.0.0',
        port=5000,
        threaded=True # Memungkinkan beberapa request ditangani secara bersamaan oleh development server
    )