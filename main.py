from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import uuid
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
DISTANCE_METRICS = {
    'VGG-Face': 'cosine',
    'Facenet': 'cosine', 
    'Facenet512': 'cosine',
    'OpenFace': 'cosine',
    'DeepFace': 'cosine',
    'DeepID': 'cosine',
    'ArcFace': 'cosine',
    'Dlib': 'euclidean',
    'SFace': 'cosine'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image(image_path, max_size=(800, 800), quality=85):
    """Optimize image for faster processing"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save optimized image
            optimized_path = image_path.replace('.', '_optimized.')
            img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
            
            return optimized_path
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        return image_path

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of supported models"""
    return jsonify({
        'models': SUPPORTED_MODELS,
        'default': 'VGG-Face'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    files_to_delete = []
    
    try:
        # Validate request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Please upload both images'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']
        model_name = request.form.get('model', 'VGG-Face')

        # Validate files
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'File names cannot be empty'}), 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({'error': 'Unsupported file format. Please use JPG, PNG, GIF, BMP, or WebP'}), 400

        # Validate model
        if model_name not in SUPPORTED_MODELS:
            return jsonify({'error': f'Unsupported model. Available models: {list(SUPPORTED_MODELS.keys())}'}), 400

        # Generate unique filenames
        ext1 = os.path.splitext(file1.filename)[1].lower()
        ext2 = os.path.splitext(file2.filename)[1].lower()
        
        filename1 = f"{uuid.uuid4()}{ext1}"
        filename2 = f"{uuid.uuid4()}{ext2}"
        
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        # Save uploaded files
        file1.save(img1_path)
        file2.save(img2_path)
        files_to_delete.extend([img1_path, img2_path])

        logger.info(f"Processing images with model: {model_name}")

        # Optimize images for faster processing
        try:
            optimized_img1 = optimize_image(img1_path)
            optimized_img2 = optimize_image(img2_path)
            
            if optimized_img1 != img1_path:
                files_to_delete.append(optimized_img1)
            if optimized_img2 != img2_path:
                files_to_delete.append(optimized_img2)
                
            # Use optimized images for processing
            process_img1 = optimized_img1
            process_img2 = optimized_img2
            
        except Exception as e:
            logger.warning(f"Image optimization failed, using original images: {e}")
            process_img1 = img1_path
            process_img2 = img2_path

        # Get appropriate distance metric for the model
        distance_metric = DISTANCE_METRICS.get(model_name, 'cosine')

        # Perform face verification with enhanced error handling
        try:
            result = DeepFace.verify(
                img1_path=process_img1,
                img2_path=process_img2,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=False,  # Allow processing even if face detection is uncertain
                detector_backend='opencv'  # Use OpenCV as it's more reliable
            )
            
            logger.info(f"Verification successful with {model_name}")
            
            # Enhanced response with additional information
            response_data = {
                'verified': result['verified'],
                'distance': float(result['distance']),  # Ensure it's JSON serializable
                'threshold': float(result['threshold']),
                'model': result['model'],
                'similarity_metric': result['similarity_metric'],
                'confidence_score': float(1 - (result['distance'] / result['threshold'])) if result['threshold'] > 0 else 0,
                'model_description': SUPPORTED_MODELS.get(model_name, 'Unknown model'),
                'processing_info': {
                    'distance_metric': distance_metric,
                    'detection_backend': 'opencv'
                }
            }
            
            return jsonify(response_data)

        except Exception as deepface_error:
            error_msg = str(deepface_error).lower()
            
            # Handle specific DeepFace errors
            if 'face could not be detected' in error_msg:
                return jsonify({
                    'error': 'Face could not be detected in one or both images. Please ensure faces are clearly visible and well-lit.',
                    'suggestion': 'Try using different images with clearer faces, or try the OpenFace or Dlib models which are more tolerant.'
                }), 400
                
            elif 'could not find any face' in error_msg:
                return jsonify({
                    'error': 'No faces found in the uploaded images.',
                    'suggestion': 'Please upload images that clearly contain human faces.'
                }), 400
                
            elif 'dimension' in error_msg or 'shape' in error_msg:
                return jsonify({
                    'error': 'Image processing error. The images might be corrupted or in an unsupported format.',
                    'suggestion': 'Try uploading different images or converting them to JPG format.'
                }), 400
                
            elif 'memory' in error_msg or 'out of memory' in error_msg:
                return jsonify({
                    'error': 'Processing failed due to memory constraints. Try using smaller images.',
                    'suggestion': 'Resize your images to be smaller (under 1MB) and try again.'
                }), 500
                
            else:
                logger.error(f"DeepFace processing error with {model_name}: {deepface_error}")
                return jsonify({
                    'error': f'Face processing failed: {str(deepface_error)}',
                    'suggestion': f'Try using a different model. Current model: {model_name}'
                }), 500

    except Exception as general_error:
        logger.error(f"General error in predict endpoint: {general_error}")
        return jsonify({
            'error': 'An unexpected error occurred during processing.',
            'details': str(general_error)
        }), 500
        
    finally:
        # Cleanup: Delete all temporary files
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted temporary file: {file_path}")
                except Exception as delete_error:
                    logger.warning(f"Could not delete file {file_path}: {delete_error}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'supported_models': list(SUPPORTED_MODELS.keys()),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_file_size': '16MB'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB per image.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Print startup information
    print("🚀 Starting Enhanced Face Similarity Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 Supported models: {', '.join(SUPPORTED_MODELS.keys())}")
    print(f"📏 Max file size: 16MB")
    print("🌐 Server will be available at: http://localhost:5000")
    print("\n" + "="*50)
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',  # Allow connections from other devices on network
        port=5000,
        threaded=True  # Enable threading for better performance
    )