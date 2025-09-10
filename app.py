# Import modul yang diperlukan dari Flask dan pustaka lainnya
from flask import Flask, request, jsonify, url_for, send_from_directory, render_template
import os
import glob
from ultralytics import YOLO # Pustaka untuk model deteksi objek YOLO
from werkzeug.utils import secure_filename # Untuk mengamankan nama file yang diunggah
import uuid # Untuk menghasilkan ID unik
import logging # Untuk logging aplikasi
import json # Untuk bekerja dengan data JSON
from datetime import datetime # Untuk informasi waktu (opsional, tidak digunakan secara langsung di metadata saat ini)
import shutil # Untuk operasi file seperti menghapus direktori
import cv2  # Import OpenCV untuk manipulasi video

# Inisialisasi aplikasi Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Konfigurasi folder untuk unggahan dan hasil
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Folder untuk menyimpan video asli yang diunggah
app.config['RESULT_FOLDER'] = 'static/results' # Folder untuk menyimpan video hasil deteksi
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'} # Ekstensi file video yang diizinkan

# Setup logging untuk aplikasi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat direktori jika belum ada, untuk memastikan folder tersedia
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Inisialisasi model YOLO
# Pastikan file model 'yolo11n.pt' ada di root proyek Anda
try:
    model = YOLO('yolo11n.pt') # Memuat model YOLO
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise # Hentikan aplikasi jika model gagal dimuat

# Fungsi pembantu untuk memeriksa apakah ekstensi file diizinkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi pembantu untuk mendapatkan tipe MIME berdasarkan ekstensi file
def get_mime_type(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    mime_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime'
    }
    return mime_types.get(ext, 'video/mp4') # Default ke video/mp4 jika tidak ditemukan

# Fungsi untuk mengekstrak metadata dari hasil deteksi YOLO
def get_video_metadata(results):
    metadata = {
        'labels': [], # Daftar label objek yang terdeteksi
        'detection_count': 0, # Total jumlah deteksi
        'average_confidence': 0.0 # Rata-rata tingkat kepercayaan deteksi
    }
    # YOLO results can be a list of Results objects, each containing detections for a frame.
    # We need to iterate through all results and all boxes within each result to get the total count.
    
    # To get a *total* count across the entire video, we'll sum up detections from all frames.
    # If you want a count per frame, the logic would be different (e.g., store a list of counts).
    # For simplicity, this function currently aggregates all detections across all frames.
    
    confidences = [] # Daftar untuk menyimpan nilai kepercayaan
    unique_labels = set() # Menggunakan set untuk menyimpan label unik

    for result in results: # Iterasi setiap objek hasil (biasanya per frame)
        if hasattr(result, 'boxes') and result.boxes: # Pastikan ada kotak deteksi
            for box in result.boxes: # Iterasi setiap kotak deteksi dalam frame
                if box.cls is not None and box.conf is not None: # Pastikan kelas dan kepercayaan ada
                    label = model.names.get(int(box.cls), 'unknown') # Dapatkan nama label dari ID kelas
                    confidence = float(box.conf) # Dapatkan nilai kepercayaan
                    unique_labels.add(label) # Tambahkan label ke set unik
                    confidences.append(confidence)
                    metadata['detection_count'] += 1 # Tambahkan ke total hitungan deteksi

    # Urutkan dan konversi set label unik ke list
    metadata['labels'] = sorted(list(unique_labels))
    if confidences:
        # Hitung rata-rata kepercayaan jika ada deteksi
        metadata['average_confidence'] = sum(confidences) / len(confidences)
    
    return metadata

# Rute utama untuk menampilkan halaman index.html
@app.route('/')
def index():
    # Ambil daftar video yang sudah diproses untuk ditampilkan di halaman awal
    result_dirs = glob.glob(os.path.join(app.config['RESULT_FOLDER'], 'result_*'))
    processed_videos = []
    for dir_path in result_dirs:
        # Cari file video di dalam setiap direktori hasil
        # Kita sekarang mencari video dengan awalan 'annotated_'
        videos = glob.glob(os.path.join(dir_path, "annotated_*.mp4"))
        # Jika tidak ada video annotated, coba cari video asli hasil YOLO
        if not videos:
             videos = glob.glob(os.path.join(dir_path, "*.mp4"))

        metadata_path = os.path.join(dir_path, 'metadata.json')
        video_id = os.path.basename(dir_path).replace('result_', '')
        
        metadata = {'labels': [], 'detection_count': 0, 'average_confidence': 0.0}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f) # Muat metadata dari file JSON
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_path}: {e}")
        
        for video in videos:
            relative_path = os.path.relpath(video, app.static_folder)
            original_filename = metadata.get('original_filename', 'Unknown')
            processed_videos.append({
                'id': video_id,
                'url': url_for('static', filename=relative_path, _external=True),
                'original_url': metadata.get('original_url', ''),
                'result_url': url_for('static', filename=relative_path, _external=True),
                'original_filename': original_filename,
                'processed_file_mime_type': get_mime_type(video),
                'type': 'video',
                'metadata': metadata
            })
    
    return render_template('index.html', processed_videos=processed_videos)

# Rute API untuk mendapatkan daftar video yang sudah diproses
@app.route('/videos', methods=['GET'])
def get_videos():
    try:
        result_dirs = glob.glob(os.path.join(app.config['RESULT_FOLDER'], 'result_*'))
        processed_videos = []
        for dir_path in result_dirs:
            # Kita sekarang mencari video dengan awalan 'annotated_'
            videos = glob.glob(os.path.join(dir_path, "annotated_*.mp4"))
            # Jika tidak ada video annotated, coba cari video asli hasil YOLO
            if not videos:
                videos = glob.glob(os.path.join(dir_path, "*.mp4"))

            metadata_path = os.path.join(dir_path, 'metadata.json')
            video_id = os.path.basename(dir_path).replace('result_', '')
            
            metadata = {'labels': [], 'detection_count': 0, 'average_confidence': 0.0}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_path}: {e}")
            
            for video in videos:
                relative_path = os.path.relpath(video, app.static_folder)
                original_filename = metadata.get('original_filename', 'Unknown')
                processed_videos.append({
                    'id': video_id,
                    'url': url_for('static', filename=relative_path, _external=True),
                    'original_url': metadata.get('original_url', ''),
                    'result_url': url_for('static', filename=relative_path, _external=True),
                    'original_filename': original_filename,
                    'processed_file_mime_type': get_mime_type(video),
                    'type': 'video',
                    'metadata': metadata
                })
        
        return jsonify({'success': True, 'videos': processed_videos})
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute API untuk mengunggah dan memproses video
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        # Periksa apakah ada file video dalam permintaan
        if 'video' not in request.files:
            logger.warning("No video file provided in request")
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        video = request.files['video']
        # Periksa apakah nama file kosong atau ekstensi tidak diizinkan
        if video.filename == '' or not allowed_file(video.filename):
            logger.warning(f"Invalid file or no file selected: {video.filename}")
            return jsonify({'success': False, 'error': 'Invalid file or no file selected'}), 400

        # Hasilkan nama file unik dan ID untuk video
        video_id = uuid.uuid4().hex # ID unik untuk video
        original_filename_secure = secure_filename(f"{video_id}_{video.filename}") # Nama file yang aman
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename_secure) # Path lengkap file asli
        video.save(video_path) # Simpan video asli
        logger.info(f"Saved uploaded video: {video_path}")

        # Define original_url here, after saving the video
        original_url = url_for('static', filename=f'uploads/{original_filename_secure}', _external=True) #

        # Proses deteksi menggunakan model YOLO
        output_name = f'result_{video_id}' # Nama folder hasil
        output_dir = os.path.join(app.config['RESULT_FOLDER'], output_name) # Path folder hasil
        os.makedirs(output_dir, exist_ok=True) # Buat folder hasil jika belum ada
        
        # Jalankan deteksi YOLO pada video
        # save=True akan menyimpan video dengan bounding box
        # project dan name menentukan lokasi dan nama folder hasil
        # conf adalah threshold kepercayaan, imgsz adalah ukuran gambar, device='cpu' menggunakan CPU
        results = model.predict(
            source=video_path,
            save=True,
            project=app.config['RESULT_FOLDER'],
            name=output_name,
            exist_ok=True,
            conf=0.5, # Tingkat kepercayaan minimal untuk deteksi
            imgsz=640, # Ukuran gambar untuk inferensi
            device='cpu' # Gunakan CPU untuk pemrosesan
        )
        logger.info(f"YOLO processing completed for: {video_path}")

        yolo_output_video_path = None
        # YOLO typically saves results in the 'name' directory directly, or in a 'name/predict' subfolder.
        # Check results[0].path first, then try the directory if it's not a file.
        if results and hasattr(results[0], 'path') and os.path.isfile(results[0].path):
            yolo_output_video_path = results[0].path
        else:
            # Fallback: Look for the video in the output_dir or a 'predict' subfolder within it.
            # YOLO results usually have a 'save_dir' attribute pointing to the output directory.
            # This is more robust than guessing the filename.
            if results and hasattr(results[0], 'save_dir'):
                yolo_save_dir = results[0].save_dir
                detected_videos_in_dir = glob.glob(os.path.join(yolo_save_dir, "*.mp4"))
                if detected_videos_in_dir:
                    yolo_output_video_path = detected_videos_in_dir[0]
            
            if not yolo_output_video_path: # If still not found, try direct in output_dir
                detected_videos_in_dir = glob.glob(os.path.join(output_dir, "*.mp4"))
                if detected_videos_in_dir:
                    yolo_output_video_path = detected_videos_in_dir[0]

        if not yolo_output_video_path or not os.path.exists(yolo_output_video_path):
            logger.error(f"YOLO processed video file not found at expected path or in {output_dir}")
            return jsonify({'success': False, 'error': 'YOLO processed video file not found'}), 500

        # Ekstrak dan simpan metadata dari hasil YOLO
        metadata = get_video_metadata(results)
        metadata['original_filename'] = video.filename # Tambahkan nama file asli ke metadata
        metadata['original_url'] = original_url # Tambahkan URL file asli ke metadata
        metadata_path = os.path.join(output_dir, 'metadata.json') # Path file metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4) # Simpan metadata ke file JSON dengan indentasi

        # --- Bagian Baru: Overlay jumlah deteksi pada video menggunakan OpenCV ---
        cap = cv2.VideoCapture(yolo_output_video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open YOLO processed video file: {yolo_output_video_path}")
            return jsonify({'success': False, 'error': 'Could not open processed video for annotation'}), 500

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Nama file untuk video yang sudah dianotasi
        output_annotated_filename = f"annotated_{os.path.basename(yolo_output_video_path)}"
        output_annotated_path = os.path.join(output_dir, output_annotated_filename)
        
        # Codec untuk menyimpan video (mp4v untuk .mp4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_annotated_path, fourcc, fps, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)  # Putih
        text_background_color = (0, 0, 0)  # Hitam
        padding = 10

        # Teks yang akan ditampilkan (menggunakan total deteksi dari metadata)
        text_to_display = f"Total Kendaraan: {metadata['detection_count']}"
        
        # Hitung ukuran teks untuk latar belakang
        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
        
        # Posisi teks (di pojok kiri atas)
        text_x = 10
        text_y = 10 + text_height # Memberi sedikit padding dari atas

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Gambar latar belakang untuk teks
            cv2.rectangle(frame, (text_x - padding, text_y - text_height - padding), 
                          (text_x + text_width + padding, text_y + baseline + padding), 
                          text_background_color, -1) # -1 untuk mengisi persegi
            
            # Tambahkan teks ke frame
            cv2.putText(frame, text_to_display, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            
            out.write(frame) # Tulis frame yang sudah dianotasi ke video output

        cap.release() # Lepaskan objek VideoCapture
        out.release() # Lepaskan objek VideoWriter
        logger.info(f"Annotated video saved to: {output_annotated_path}")
        # --- Akhir Bagian Baru ---

        # URL yang akan dikembalikan ke frontend sekarang menunjuk ke video yang sudah dianotasi
        result_url_annotated = url_for('static', filename=f'results/{output_name}/{output_annotated_filename}', _external=True)
        # original_url is already defined above

        logger.info(f"Processed video found: {result_url_annotated}, Metadata: {metadata}")
        return jsonify({
            'success': True,
            'id': video_id,
            'original_url': original_url,
            'result_url': result_url_annotated, # Menggunakan URL video yang sudah dianotasi
            'original_filename': video.filename,
            'processed_file_mime_type': get_mime_type(output_annotated_filename),
            'type': 'video',
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute API untuk memperbarui metadata video (PUT request)
@app.route('/videos/<video_id>', methods=['PUT'])
def update_video_metadata(video_id):
    try:
        data = request.get_json() # Ambil data JSON dari permintaan
        # Validasi data yang diterima
        if not data or 'labels' not in data or 'detection_count' not in data or 'average_confidence' not in data:
            logger.warning(f"Invalid metadata update request for video_id: {video_id}")
            return jsonify({'success': False, 'error': 'Invalid metadata provided'}), 400

        result_dir = os.path.join(app.config['RESULT_FOLDER'], f'result_{video_id}')
        metadata_path = os.path.join(result_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found for video_id: {video_id}")
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # Muat metadata yang sudah ada
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Perbarui metadata dengan data baru
        metadata['labels'] = data['labels']
        metadata['detection_count'] = data['detection_count']
        metadata['average_confidence'] = data['average_confidence']

        # Simpan metadata yang sudah diperbarui
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Updated metadata for video_id: {video_id}")
        return jsonify({'success': True, 'metadata': metadata})

    except Exception as e:
        logger.error(f"Error updating metadata for video_id {video_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute API untuk menghapus video dan hasilnya (DELETE request)
@app.route('/videos/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    try:
        result_dir = os.path.join(app.config['RESULT_FOLDER'], f'result_{video_id}')
        if not os.path.exists(result_dir):
            logger.warning(f"Result directory not found for video_id: {video_id}")
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # Cari file asli yang diunggah untuk dihapus juga
        metadata_path = os.path.join(result_dir, 'metadata.json')
        original_filename = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                original_filename = metadata.get('original_filename') # Ambil nama file asli

        # Hapus direktori hasil deteksi beserta isinya
        shutil.rmtree(result_dir)
        logger.info(f"Deleted result directory: {result_dir}")

        # Hapus file asli jika ditemukan
        if original_filename:
            # Gunakan secure_filename untuk merekonstruksi nama file yang aman
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{video_id}_{original_filename}"))
            if os.path.exists(original_path):
                os.remove(original_path)
                logger.info(f"Deleted original file: {original_path}")

        return jsonify({'success': True, 'message': f'Video {video_id} deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting video_id {video_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Rute untuk menyajikan file statis (CSS, JS, gambar, video)
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({'success': False, 'error': 'File not found'}), 404

# Jalankan aplikasi Flask
if __name__ == '__main__':
    # debug=True akan mengaktifkan mode debug (reload otomatis, pesan error detail)
    # host='0.0.0.0' membuat server dapat diakses dari luar localhost
    # port=5000 adalah port default Flask
    app.run(debug=True, host='0.0.0.0', port=5000)