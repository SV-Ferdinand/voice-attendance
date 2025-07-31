from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sqlite3
import numpy as np
import librosa
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from datetime import datetime
import jwt
import bcrypt
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)

# Konfigurasi
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'data/voice_samples'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Inisialisasi Voice Encoder
voice_encoder = VoiceEncoder()

# Pastikan folder ada
os.makedirs('data/voice_samples', exist_ok=True)
os.makedirs('data/database', exist_ok=True)
os.makedirs('data/models', exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('data/database/attendance.db')
    cursor = conn.cursor()
    
    # Tabel siswa
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            voice_registered BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabel admin
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabel absensi
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT DEFAULT 'HADIR',
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    # Tabel voice embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_embeddings (
            student_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# API Routes
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    admin_id = data.get('id')
    admin_name = data.get('name')
    
    # Validasi input
    if not admin_id or len(admin_id) != 9 or not admin_name:
        return jsonify({'error': 'Data tidak valid'}), 400
    
    # Untuk demo, kita terima semua admin
    # Di production, cek ke database
    token = jwt.encode({
        'admin_id': admin_id,
        'admin_name': admin_name,
        'exp': datetime.utcnow().timestamp() + 3600  # 1 jam
    }, app.config['SECRET_KEY'])
    
    return jsonify({
        'success': True,
        'token': token,
        'admin_name': admin_name
    })

@app.route('/api/students', methods=['GET', 'POST'])
def handle_students():
    if request.method == 'GET':
        conn = sqlite3.connect('data/database/attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM students ORDER BY created_at DESC')
        students = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': s[0],
            'name': s[1],
            'class': s[2],
            'voice_registered': bool(s[3])
        } for s in students])
    
    elif request.method == 'POST':
        data = request.get_json()
        student_id = data.get('id')
        name = data.get('name')
        class_name = data.get('class')
        
        if not student_id or len(student_id) != 9 or not name or not class_name:
            return jsonify({'error': 'Data tidak valid'}), 400
        
        try:
            conn = sqlite3.connect('data/database/attendance.db')
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO students (id, name, class) VALUES (?, ?, ?)',
                (student_id, name, class_name)
            )
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Siswa berhasil didaftarkan'})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'ID siswa sudah terdaftar'}), 400

@app.route('/api/voice/register', methods=['POST'])
def register_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'File audio tidak ditemukan'}), 400
    
    audio_file = request.files['audio']
    student_id = request.form.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'Student ID diperlukan'}), 400
    
    try:
        # Simpan file audio sementara
        filename = secure_filename(f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Proses audio dengan Resemblyzer
        wav = preprocess_wav(filepath)
        embedding = voice_encoder.embed_utterance(wav)
        
        # Simpan embedding ke database
        conn = sqlite3.connect('data/database/attendance.db')
        cursor = conn.cursor()
        
        # Convert embedding ke binary
        embedding_binary = embedding.tobytes()
        
        cursor.execute('''
            INSERT OR REPLACE INTO voice_embeddings (student_id, embedding) 
            VALUES (?, ?)
        ''', (student_id, embedding_binary))
        
        # Update status voice_registered
        cursor.execute(
            'UPDATE students SET voice_registered = TRUE WHERE id = ?',
            (student_id,)
        )
        
        conn.commit()
        conn.close()
        
        # Hapus file sementara
        os.remove(filepath)
        
        return jsonify({'success': True, 'message': 'Voice berhasil diregistrasi'})
        
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/api/voice/verify', methods=['POST'])
def verify_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'File audio tidak ditemukan'}), 400
    
    audio_file = request.files['audio']
    student_id = request.form.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'Student ID diperlukan'}), 400
    
    try:
        # Simpan file audio sementara
        filename = secure_filename(f"verify_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Proses audio
        wav = preprocess_wav(filepath)
        new_embedding = voice_encoder.embed_utterance(wav)
        
        # Ambil embedding yang sudah tersimpan
        conn = sqlite3.connect('data/database/attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT embedding FROM voice_embeddings WHERE student_id = ?', (student_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            os.remove(filepath)
            return jsonify({'error': 'Voice belum diregistrasi'}), 400
        
        # Convert binary ke numpy array
        stored_embedding = np.frombuffer(result[0], dtype=np.float32)
        
        # Hitung similarity
        similarity = np.dot(new_embedding, stored_embedding) / (
            np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
        )
        
        # Threshold untuk verifikasi (bisa disesuaikan)
        threshold = 0.7
        is_verified = similarity > threshold
        
        if is_verified:
            # Catat kehadiran
            now = datetime.now()
            cursor.execute('''
                INSERT INTO attendance (student_id, date, time) 
                VALUES (?, ?, ?)
            ''', (student_id, now.date(), now.time()))
            conn.commit()
        
        conn.close()
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'verified': is_verified,
            'similarity': float(similarity),
            'message': 'Verifikasi berhasil' if is_verified else 'Verifikasi gagal'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error verifying voice: {str(e)}'}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date = request.args.get('date', datetime.now().date())
    
    conn = sqlite3.connect('data/database/attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT a.student_id, s.name, s.class, a.date, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (date,))
    
    records = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'student_id': r[0],
        'name': r[1],
        'class': r[2],
        'date': r[3],
        'time': r[4],
        'status': r[5]
    } for r in records])

@app.route('/api/speech/recognize', methods=['POST'])
def recognize_speech():
    """Endpoint untuk speech recognition menggunakan SpeechBrain"""
    if 'audio' not in request.files:
        return jsonify({'error': 'File audio tidak ditemukan'}), 400
    
    try:
        audio_file = request.files['audio']
        
        # Simpan file sementara
        filename = secure_filename(f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Load audio dengan librosa
        audio, sr = librosa.load(filepath, sr=16000)
        
        # Placeholder untuk speech recognition
        # Di implementasi nyata, gunakan SpeechBrain model
        recognized_text = "HADIR"  # Simulasi hasil
        confidence = 0.95
        
        # Hapus file sementara
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'text': recognized_text,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': f'Error recognizing speech: {str(e)}'}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)