from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename
import traceback
from gnn_fraud_model import process_fraud_detection, MinimalFraudDetectionSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'GNN Fraud Detection API is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process fraud detection"""
    try:
        logger.info("Received file upload request")
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV, XLS, or XLSX files.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved to {file_path}")
        
        # Read file based on extension
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Validate data
        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400
        
        if df.shape[0] < 10:
            return jsonify({'error': 'Dataset too small. Please upload a file with at least 10 rows.'}), 400
        
        # Process fraud detection
        logger.info("Starting fraud detection processing...")
        
        try:
            # Save model path
            model_path = os.path.join(app.config['RESULTS_FOLDER'], f'model_{filename.split(".")[0]}.pth')
            
            # Process the data
            viz_data, fraud_detector = process_fraud_detection(file_path, model_path)
            
            # Save results
            results_file = os.path.join(app.config['RESULTS_FOLDER'], f'results_{filename.split(".")[0]}.json')
            with open(results_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            
            logger.info("Fraud detection processing completed successfully")
            
            # Return visualization data
            return jsonify({
                'success': True,
                'message': 'File processed successfully',
                'data': viz_data,
                'filename': filename,
                'original_shape': df.shape
            })
            
        except Exception as e:
            logger.error(f"Error in fraud detection processing: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f'Error processing fraud detection: {str(e)}',
                'details': 'Please check if your data has numerical features suitable for fraud detection.'
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Return sample data for testing"""
    try:
        sample_file = 'sample_data.csv'
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            return jsonify({
                'success': True,
                'sample_data': df.head(10).to_dict('records'),
                'shape': df.shape,
                'columns': df.columns.tolist()
            })
        else:
            return jsonify({'error': 'Sample data file not found'}), 404
    
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return jsonify({'error': f'Error loading sample data: {str(e)}'}), 500

@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    """Get saved results by filename"""
    try:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f'results_{filename}.json')
        
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return jsonify({
            'success': True,
            'data': results,
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({'error': f'Error loading results: {str(e)}'}), 500

@app.route('/list-results', methods=['GET'])
def list_results():
    """List all available results"""
    try:
        results_files = []
        results_dir = app.config['RESULTS_FOLDER']
        
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.startswith('results_') and file.endswith('.json'):
                    filename = file.replace('results_', '').replace('.json', '')
                    file_path = os.path.join(results_dir, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)
                    
                    results_files.append({
                        'filename': filename,
                        'size': file_size,
                        'modified': modified_time
                    })
        
        return jsonify({
            'success': True,
            'results': results_files
        })
    
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return jsonify({'error': f'Error listing results: {str(e)}'}), 500

@app.route('/validate-data', methods=['POST'])
def validate_data():
    """Validate uploaded data before processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read first few rows to validate
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        file.save(temp_path)
        
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(temp_path, nrows=100)
            else:
                df = pd.read_excel(temp_path, nrows=100)
            
            # Basic validation
            validation_results = {
                'valid': True,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Check for minimum requirements
            if df.shape[0] < 10:
                validation_results['valid'] = False
                validation_results['error'] = 'Dataset too small (minimum 10 rows required)'
            
            if len(validation_results['numeric_columns']) < 2:
                validation_results['valid'] = False
                validation_results['error'] = 'Insufficient numeric features (minimum 2 required)'
            
            # Clean up temp file
            os.remove(temp_path)
            
            return jsonify(validation_results)
            
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return jsonify({'error': f'Error validating data: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({'error': 'Resource not found.'}), 404

if __name__ == '__main__':
    print("Starting GNN Fraud Detection API...")
    print("Available endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /upload           - Upload and process file")
    print("  GET  /sample-data      - Get sample data")
    print("  POST /validate-data    - Validate uploaded data")
    print("  GET  /results/<filename> - Get results by filename")
    print("  GET  /list-results     - List all results")
    print("\nServer starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)