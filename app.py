# Import modul yang diperlukan dari Flask dan pustaka lainnya
from flask import Flask, request, jsonify, url_for, send_from_directory, render_template, Response
import os
import glob
from ultralytics import YOLO # Pustaka untuk model deteksi objek YOLO
from werkzeug.utils import secure_filename # Untuk mengamankan nama file yang diunggah
import uuid # Untuk menghasilkan ID unik
import logging # Untuk logging aplikasi
import json # Untuk bekerja dengan data JSON
from datetime import datetime # Untuk informasi waktu
import shutil # Untuk operasi file seperti menghapus direktori
import cv2  # Import OpenCV untuk manipulasi video (digunakan untuk mock dan anotasi dasar)
import time # Untuk simulasi streaming (tidak digunakan lagi, tapi tetap dipertahankan)
import numpy as np # Untuk operasi array (digunakan untuk mock)

# Inisialisasi aplikasi Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Konfigurasi folder untuk unggahan dan hasil
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Folder untuk menyimpan video asli yang diunggah
# Perhatikan: Folder hasil YOLO akan disimpan di 'runs/detect/...' secara default oleh Ultralytics,
# tapi kita akan menggunakan 'static/results' sebagai project folder agar mudah diakses.
app.config['RESULT_BASE_FOLDER'] = 'static/results' 
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'} # Ekstensi file video yang diizinkan

# Setup logging untuk aplikasi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat direktori jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_BASE_FOLDER'], exist_ok=True)

# Inisialisasi model YOLO
MODEL_PATH = 'Model/yolo11n.pt'
model = None
try:
    if os.path.exists(MODEL_PATH):
        # Memuat model YOLO
        model = YOLO(MODEL_PATH) 
        logger.info("YOLO model loaded successfully.")
    else:
        logger.warning(f"YOLO model file not found at {MODEL_PATH}. Detection functionality will be disabled.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

# --- Fungsi Pembantu ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_mime_type(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    mime_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime'
    }
    return mime_types.get(ext, 'video/mp4')

def get_video_metadata(results):
    """
    Ekstrak metadata deteksi dari hasil YOLOv8/YOLOv11 (Ultralytics).
    Akan diadaptasi untuk YOLOv11 jika struktur hasilnya sama dengan YOLOv8.
    """
    metadata = {
        'labels': {}, # Dictionary untuk menyimpan hitungan per label
        'detection_count': 0, 
        'average_confidence': 0.0,
        'frame_count': 0
    }
    confidences = []
    
    # Loop melalui setiap frame dalam hasil
    for result in results:
        metadata['frame_count'] += 1
        if hasattr(result, 'boxes') and result.boxes:
            # Iterasi setiap kotak deteksi dalam frame
            for box in result.boxes:
                if box.cls is not None and box.conf is not None:
                    # Ambil ID kelas dan nama label
                    class_id = int(box.cls[0].item()) # Ambil nilai skalar dari tensor
                    confidence = float(box.conf[0].item())

                    # Pastikan model.names ada jika model dimuat
                    label = model.names.get(class_id, f'Class_{class_id}') if model and hasattr(model, 'names') else f'Class_{class_id}'
                    
                    # Update statistik
                    metadata['detection_count'] += 1
                    confidences.append(confidence)
                    
                    # Hitung distribusi kelas
                    if label not in metadata['labels']:
                        metadata['labels'][label] = 0
                    metadata['labels'][label] += 1

    if confidences:
        metadata['average_confidence'] = sum(confidences) / len(confidences)
    
    # Konversi dictionary labels ke list untuk frontend (jika diperlukan)
    label_distribution = [{'name': k, 'count': v} for k, v in metadata['labels'].items()]
    metadata['labels_distribution'] = sorted(label_distribution, key=lambda x: x['count'], reverse=True)
    
    # Bersihkan key 'labels' yang hanya berupa hitungan untuk menghindari kebingungan
    del metadata['labels']
    
    return metadata

# --- Rute Flask ---

@app.route('/')
def index():
    """Rute utama untuk menampilkan halaman index.html."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Rute API untuk mengunggah dan memproses video menggunakan YOLO."""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        video = request.files['video']
        if video.filename == '' or not allowed_file(video.filename):
            return jsonify({'success': False, 'error': 'Invalid file or no file selected'}), 400

        video_id = uuid.uuid4().hex
        # Gunakan nama file asli + ID unik untuk mencegah konflik
        original_filename_secure = secure_filename(f"{video_id}_{video.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename_secure)
        video.save(video_path)
        logger.info(f"Saved uploaded video: {video_path}")

        original_url = url_for('static', filename=f'uploads/{original_filename_secure}', _external=True)

        if model:
            logger.info("Starting YOLO processing...")
            
            # Tentukan direktori output untuk YOLO
            # Ultralytics akan menyimpan hasil di runs/detect/result_video_id
            output_name = f'result_{video_id}'
            
            # Hapus folder lama jika ada (untuk debugging)
            output_dir = os.path.join(app.config['RESULT_BASE_FOLDER'], output_name)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            results = model.predict(
                source=video_path,
                save=True, # Simpan video dengan kotak deteksi
                project=app.config['RESULT_BASE_FOLDER'],
                name=output_name,
                exist_ok=True,
                conf=0.25, # Mengatur confidence lebih rendah untuk hasil yang lebih baik
                imgsz=640,
                device='cpu'
            )
            logger.info(f"YOLO processing completed for: {video_path}")

            # Cari path video output YOLO. Ultralytics menyimpannya di subfolder.
            # Kita perlu mencari file .mp4 atau .avi di dalam direktori output_dir/nama_file_video
            
            # Path yang diharapkan oleh Ultralytics: runs/detect/project_name/nama_video_asli.mp4
            # Dalam kasus ini: static/results/result_video_id/nama_video_asli.mp4
            detected_videos = glob.glob(os.path.join(output_dir, os.path.splitext(os.path.basename(video.filename))[0] + '.*'))
            
            yolo_output_video_path = None
            if detected_videos:
                # Ambil video pertama yang ditemukan
                yolo_output_video_path = detected_videos[0]
                
            if not yolo_output_video_path or not os.path.exists(yolo_output_video_path):
                logger.error(f"YOLO processed video file not found in {output_dir}")
                # Coba cari semua file video yang dihasilkan
                detected_videos_in_dir = glob.glob(os.path.join(output_dir, "*"))
                if detected_videos_in_dir:
                    yolo_output_video_path = next((f for f in detected_videos_in_dir if allowed_file(f)), None)
                
                if not yolo_output_video_path:
                    logger.error("Failed to find any processed video file.")
                    return jsonify({'success': False, 'error': 'YOLO processed video file not found'}), 500

            # URL untuk video hasil
            result_file_name = os.path.basename(yolo_output_video_path)
            result_url = url_for('static', filename=f'results/{output_name}/{result_file_name}', _external=True)

            metadata = get_video_metadata(results)
            logger.info(f"Processed video found: {result_url}, Metadata: {metadata}")
            
            return jsonify({
                'success': True,
                'id': video_id,
                'original_url': original_url,
                'result_url': result_url,
                'original_filename': video.filename,
                'processed_file_mime_type': get_mime_type(result_file_name),
                'type': 'video',
                'metadata': metadata
            })
        else:
            # Jika model tidak dimuat (misalnya, file yolo11n.pt tidak ada), 
            # kembalikan respons sederhana untuk unggahan tanpa deteksi
            logger.warning("YOLO model not loaded. Skipping video processing and returning original URL.")
            return jsonify({
                'success': True,
                'id': video_id,
                'original_url': original_url,
                'result_url': original_url, # Kembalikan URL asli jika tidak diproses
                'original_filename': video.filename,
                'processed_file_mime_type': get_mime_type(video.filename),
                'type': 'video',
                'metadata': {'labels_distribution': [{'name': 'Unprocessed', 'count': 0}], 'detection_count': 0, 'average_confidence': 0.0}
            })

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute API untuk memperbarui metadata video (DELETE request) - Biarkan tetap seperti sebelumnya untuk menjaga fitur
@app.route('/videos/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    try:
        # Hapus direktori hasil (mengandung video yang diproses)
        result_dir = os.path.join(app.config['RESULT_BASE_FOLDER'], f'result_{video_id}')
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            logger.info(f"Deleted result directory: {result_dir}")

        # Hapus file video asli
        upload_dir = app.config['UPLOAD_FOLDER']
        # Cari file di folder upload berdasarkan prefix ID
        original_files = glob.glob(os.path.join(upload_dir, f"{video_id}_*"))
        for f in original_files:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"Deleted original file: {f}")
        
        if not result_dir and not original_files:
            logger.warning(f"Video assets not found for video_id: {video_id}")
            return jsonify({'success': False, 'error': 'Video assets not found'}), 404

        return jsonify({'success': True, 'message': f'Video {video_id} deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting video_id {video_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute untuk menyajikan file statis (CSS, JS, gambar, video) - Biarkan tetap seperti sebelumnya
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        # Menangani path yang mungkin berada di dalam sub-folder hasil YOLO
        if filename.startswith('results/'):
            # Misalnya: results/result_id/video.mp4
            return send_from_directory(app.static_folder, filename)
        
        # Atau folder lain di static/
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({'success': False, 'error': 'File not found'}), 404
        
# --- Rute Mock untuk Simulasi Deteksi Real-time ---
# Rute ini dipertahankan sebagai MOCK, tetapi difokuskan pada penggunaan Webcam/Video sampel di frontend.

def generate_mock_detections_stream(video_path):
    """
    Simulasi generator untuk streaming bingkai video yang disajikan dengan deteksi mock.
    Dalam aplikasi nyata, ini akan membaca bingkai, menjalankan YOLO, dan mengembalikan bingkai beranotasi.
    """
    # Menggunakan video dummy atau placeholder sederhana untuk mematuhi aturan platform
    while True:
        # Buat bingkai hitam sederhana 640x360 dengan teks
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        text = "YOLO Mock Stream: Upload Video Anda"
        cv2.putText(frame, text, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1) # Simulasi 10 FPS

@app.route('/video_feed')
def video_feed():
    """Rute ini mengalirkan bingkai video simulasi untuk dashboard."""
    # Karena kita fokus pada upload, rute ini sekarang mengalirkan placeholder
    # untuk mencegah pemrosesan video yang lambat saat memuat halaman utama.
    return Response(generate_mock_detections_stream(None), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    # Di lingkungan Canvas, biarkan host dan port default
    app.run(debug=True)
