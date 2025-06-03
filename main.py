from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import uuid
from PIL import Image, UnidentifiedImageError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    'Dlib': 'Traditional computer vision approach (often good for less clear faces)',
    'SFace': 'Lightweight and fast model'
}

# Available distance metrics and their descriptions
AVAILABLE_DISTANCE_METRICS = {
    'cosine': 'Cosine Similarity (commonly used)',
    'euclidean': 'Euclidean Distance',
    'euclidean_l2': 'Euclidean L2-norm (squared Euclidean, often more sensitive to large differences)'
}

# Available detector backends and their descriptions
AVAILABLE_DETECTOR_BACKENDS = {
    'opencv': 'OpenCV Haar Cascade (fast, baseline)',
    'ssd': 'SSD (Single Shot MultiBox Detector - good balance)',
    'dlib': 'Dlib HoG (good for frontal faces)',
    'mtcnn': 'MTCNN (Multi-task Cascaded Convolutional Networks - accurate, can be slower)',
    'retinaface': 'RetinaFace (accurate, good for various conditions)',
    'mediapipe': 'MediaPipe Face Detector (fast and robust)',
    # 'yolov8': 'YOLOv8 (if installed and supported by your DeepFace version)', # Usually needs ultralytics
    'yunet': 'YuNet (lightweight and accurate)',
    'centerface': 'CenterFace (lightweight)'
}

# Original DISTANCE_METRICS can be used for documentation or model-specific recommendations
# but user selection will take precedence.
MODEL_DEFAULT_METRICS = {
    'VGG-Face': 'cosine',
    'Facenet': 'cosine',
    'Facenet512': 'cosine',
    'OpenFace': 'euclidean_l2', # OpenFace often paired with L2
    'DeepFace': 'cosine',
    'DeepID': 'cosine',
    'ArcFace': 'cosine', # ArcFace often paired with cosine
    'Dlib': 'euclidean', # Dlib is not a deep learning model, uses different comparison logic
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
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                logger.warning(f"Image {image_path} has an unusual mode {img.mode}. Attempting RGB conversion.")
                img = img.convert('RGB')

            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            base, ext = os.path.splitext(image_path)
            optimized_path = f"{base}_optimized.jpg"
            img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
            logger.info(f"Saved optimized image to: {optimized_path} with size {img.size}")
            return optimized_path
    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file {image_path}. It might be corrupted or not an image.")
        return image_path
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        return image_path

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models_api():
    """Get list of supported models, metrics, and detectors"""
    return jsonify({
        'models': SUPPORTED_MODELS,
        'metrics': AVAILABLE_DISTANCE_METRICS,
        'detectors': AVAILABLE_DETECTOR_BACKENDS,
        'defaults': {
            'model': 'VGG-Face',
            'metric': 'cosine',
            'detector': 'opencv'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    files_to_delete = []
    request_id = str(uuid.uuid4())
    logger.info(f"[RequestID: {request_id}] Received prediction request.")

    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            logger.warning(f"[RequestID: {request_id}] Both images not found in request.")
            return jsonify({'error': 'Harap unggah kedua gambar'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # Get user choices with defaults
        model_name = request.form.get('model', 'VGG-Face')
        distance_metric = request.form.get('distance_metric', 'cosine')
        detector_backend = request.form.get('detector_backend', 'opencv')

        logger.info(f"[RequestID: {request_id}] Files: {file1.filename}, {file2.filename}. Model: {model_name}, Metric: {distance_metric}, Detector: {detector_backend}")

        if file1.filename == '' or file2.filename == '':
            logger.warning(f"[RequestID: {request_id}] One or both filenames are empty.")
            return jsonify({'error': 'Nama file tidak boleh kosong'}), 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            logger.warning(f"[RequestID: {request_id}] Unsupported file format.")
            return jsonify({'error': 'Format file tidak didukung. Harap gunakan JPG, PNG, GIF, BMP, atau WebP'}), 400

        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"[RequestID: {request_id}] Unsupported model: {model_name}")
            return jsonify({'error': f'Model tidak didukung. Model yang tersedia: {list(SUPPORTED_MODELS.keys())}'}), 400
        if distance_metric not in AVAILABLE_DISTANCE_METRICS:
            logger.warning(f"[RequestID: {request_id}] Unsupported metric: {distance_metric}")
            return jsonify({'error': f'Metrik jarak tidak didukung. Metrik yang tersedia: {list(AVAILABLE_DISTANCE_METRICS.keys())}'}), 400
        if detector_backend not in AVAILABLE_DETECTOR_BACKENDS:
            logger.warning(f"[RequestID: {request_id}] Unsupported detector: {detector_backend}")
            return jsonify({'error': f'Detector backend tidak didukung. Detector yang tersedia: {list(AVAILABLE_DETECTOR_BACKENDS.keys())}'}), 400


        ext1 = os.path.splitext(file1.filename)[1].lower() if '.' in file1.filename else '.jpg'
        ext2 = os.path.splitext(file2.filename)[1].lower() if '.' in file2.filename else '.jpg'
        filename1 = f"{request_id}_img1{ext1 if ext1 else '.tmp'}"
        filename2 = f"{request_id}_img2{ext2 if ext2 else '.tmp'}"
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(img1_path)
        file2.save(img2_path)
        files_to_delete.extend([img1_path, img2_path])
        logger.info(f"[RequestID: {request_id}] Files saved: {img1_path}, {img2_path}")

        optimized_img1 = optimize_image(img1_path)
        optimized_img2 = optimize_image(img2_path)
        if optimized_img1 != img1_path: files_to_delete.append(optimized_img1)
        if optimized_img2 != img2_path: files_to_delete.append(optimized_img2)
        
        process_img1 = optimized_img1
        process_img2 = optimized_img2
        
        enforce_detection_flag = False # User might want this as an option too in future

        logger.info(f"[RequestID: {request_id}] Using model: {model_name}, distance_metric: {distance_metric}, detector_backend: {detector_backend} for verification.")
        logger.info(f"[RequestID: {request_id}] Image paths for DeepFace: {process_img1}, {process_img2}")

        try:
            result = DeepFace.verify(
                img1_path=process_img1,
                img2_path=process_img2,
                model_name=model_name,
                distance_metric=distance_metric,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection_flag
            )
            logger.info(f"[RequestID: {request_id}] DeepFace verification result for {model_name}: Distance: {result.get('distance', -1):.4f}, Threshold: {result.get('threshold', -1):.4f}, Verified: {result.get('verified', False)}")

            similarity_percentage = 0.0
            if result.get('threshold', 0) > 0:
                similarity_percentage = (1 - (result.get('distance', float('inf')) / result['threshold'])) * 100
            elif result.get('distance', float('inf')) == 0 :
                 similarity_percentage = 100.0
            similarity_percentage = max(0.0, min(100.0, similarity_percentage))

            response_data = {
                'verified': result.get('verified', False),
                'distance': float(result.get('distance', -1)),
                'threshold': float(result.get('threshold', -1)),
                'model': result.get('model', model_name), # Actual model used by DeepFace
                'similarity_metric_used': result.get('similarity_metric', distance_metric), # Actual metric
                'similarity_percentage': round(similarity_percentage, 2),
                'facial_areas': {
                    'img1': result.get('facial_areas', {}).get('img1'),
                    'img2': result.get('facial_areas', {}).get('img2')
                },
                'model_description': SUPPORTED_MODELS.get(model_name, 'Model tidak dikenal'),
                'processing_details': {
                    'requested_model': model_name,
                    'requested_distance_metric': distance_metric,
                    'requested_detector_backend': detector_backend,
                    'enforce_detection': enforce_detection_flag,
                    'detector_backend_used': result.get('detector_backend', detector_backend) # Actual detector
                }
            }
            return jsonify(response_data)

        except Exception as deepface_error:
            error_msg_lower = str(deepface_error).lower()
            logger.error(f"[RequestID: {request_id}] DeepFace processing error: {deepface_error}", exc_info=True)
            user_error_message = f'Proses pengenalan wajah gagal: {str(deepface_error)}'
            user_suggestion = f'Coba gunakan kombinasi model/metrik/detector lain atau gambar yang berbeda. Model: {model_name}, Metrik: {distance_metric}, Detector: {detector_backend}'
            status_code = 500

            if 'face could not be detected' in error_msg_lower or "couldn't find any face" in error_msg_lower:
                user_error_message = 'Wajah tidak dapat terdeteksi pada satu atau kedua gambar.'
                user_suggestion = 'Pastikan wajah terlihat jelas dan mendapat pencahayaan yang baik. Coba detector backend yang berbeda seperti "mtcnn" atau "retinaface".'
                status_code = 400
            elif 'dimension' in error_msg_lower or 'shape' in error_msg_lower or 'corrupt' in error_msg_lower:
                user_error_message = 'Kesalahan pemrosesan gambar. Gambar mungkin rusak atau format tidak didukung.'
                user_suggestion = 'Coba unggah gambar berbeda atau konversi ke format JPG/PNG standar.'
                status_code = 400
            elif 'memory' in error_msg_lower or 'out of memory' in error_msg_lower:
                user_error_message = 'Pemrosesan gagal karena batasan memori. Coba gunakan gambar lebih kecil.'
                status_code = 500
            
            return jsonify({
                'error': user_error_message,
                'suggestion': user_suggestion,
                'details': str(deepface_error)
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
    return jsonify({
        'status': 'healthy',
        'message': 'Face Similarity API is running.',
        'supported_models_count': len(SUPPORTED_MODELS),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.errorhandler(413)
def too_large(e):
    logger.warning(f"File too large error (413): {e.description if hasattr(e, 'description') else e}")
    return jsonify({'error': 'File terlalu besar. Ukuran maksimum adalah 16MB per gambar.'}), 413

@app.errorhandler(404)
def not_found_error_handler(e): # Renamed to avoid conflict
    logger.info(f"Not found error (404) for path: {request.path}")
    return jsonify({'error': 'Endpoint tidak ditemukan'}), 404

@app.errorhandler(500)
def internal_server_error_handler(e): # Renamed to avoid conflict
    original_exception = getattr(e, 'original_exception', None)
    logger.error(f"Internal server error (500): {e}. Original Exception: {original_exception}", exc_info=True)
    return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Memulai Server Analisis Kemiripan Wajah...")
    print(f"🕒 Waktu Startup: {logging.Formatter().formatTime(logging.LogRecord(None,None,'',0,'',(),None,None), datefmt='%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Folder Unggahan: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🤖 Model yang Didukung: {', '.join(SUPPORTED_MODELS.keys())}")
    print(f"📏 Metrik Jarak Tersedia: {', '.join(AVAILABLE_DISTANCE_METRICS.keys())}")
    print(f"👁️ Detector Backend Tersedia: {', '.join(AVAILABLE_DETECTOR_BACKENDS.keys())}")
    print(f"📏 Ukuran File Maks: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"🛠️ Mode Debug Flask: {'Aktif' if app.debug else 'Nonaktif'}")
    print("🌐 Server akan tersedia di: http://localhost:5000 (atau alamat IP Anda jika host='0.0.0.0')")
    print("📝 Log akan dicatat dengan format: WAKTU - NAMA_MODUL - LEVEL - PESAN")
    print("="*60 + "\n")

    app.run(
        debug=False, # Set to False for production or when deploying to HF
        host='0.0.0.0',
        port=5000,
        threaded=True
    )