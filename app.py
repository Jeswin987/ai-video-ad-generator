
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import uuid
import time
from pathlib import Path
import json
import logging
from datetime import datetime

# Import pipeline + config
from src.video_generation_pipeline import VideoGenerationPipeline
from src.config import Config

# Initialize Flask app
app = Flask(__name__,
            static_folder="VideoGen-PWA",
            static_url_path="")
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Enable CORS for ALL routes (not just /api/*)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# WebSocket setup
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Job tracking
jobs = {}
job_lock = threading.Lock()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_job_status(job_id):
    with job_lock:
        return jobs.get(job_id)

def update_job_status(job_id, status, progress=None, message=None, video_path=None):
    with job_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        
        jobs[job_id].update({
            'status': status,
            'updated_at': datetime.now().isoformat(),
        })
        
        if progress is not None:
            jobs[job_id]['progress'] = progress
        if message is not None:
            jobs[job_id]['message'] = message
        if video_path is not None:
            jobs[job_id]['video_path'] = str(video_path)
    
    # Emit WebSocket event
    socketio.emit('job_status', {
        'job_id': job_id,
        'status': status,
        'progress': progress,
        'message': message
    }, namespace='/', room=job_id)

def generate_video_task(job_id, prompt, visual_theme, output_dir):
    """Background task for video generation"""
    try:
        logger.info(f"🎬 Starting video generation for job {job_id}")
        update_job_status(job_id, 'processing', 0, 'Initializing...')
        
        # Initialize pipeline
        pipeline = VideoGenerationPipeline()
        
        # Simulate progress stages
        stages = [
            (10, "Generating storyline..."),
            (25, "Creating scene descriptions..."),
            (40, "Generating AI images..."),
            (60, "Creating voiceovers..."),
            (75, "Assembling video clips..."),
            (90, "Finalizing video...")
        ]
        
        for progress, message in stages:
            update_job_status(job_id, 'processing', progress, message)
            time.sleep(0.5)  # Simulate work
        
        # Generate video
        video_path = pipeline.generate_video(
            prompt=prompt,
            visual_theme=visual_theme,
            output_dir=Path(output_dir)
        )
        
        if video_path and video_path.exists():
            update_job_status(job_id, 'completed', 100,
                            'Video generation complete', video_path)
            logger.info(f"✅ Job {job_id} completed successfully")
        else:
            raise RuntimeError("Video file not found after generation")
    
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        update_job_status(job_id, 'failed', 100, f'Error: {str(e)}')

# =====================================================================
# HEALTH CHECK ENDPOINT (CRITICAL FIX)
# =====================================================================

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint for connection testing"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    # Return health status
    response = jsonify({
        'status': 'online',
        'service': 'AI Video Generation API',
        'version': '1.0.0',
        'device': Config.DEVICE if hasattr(Config, 'DEVICE') else 'unknown',
        'timestamp': datetime.now().isoformat()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# =====================================================================
# API ROUTES
# =====================================================================

@app.route('/api/generate-video', methods=['POST', 'OPTIONS'])
def generate_video():
    """Start video generation job"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json(force=True)
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt'}), 400
        
        prompt = data['prompt']
        visual_theme = data.get('theme', 'Modern Cool')
        
        # Create job
        job_id = str(uuid.uuid4())
        output_dir = Config.OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with job_lock:
            jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'message': 'Job queued',
                'created_at': datetime.now().isoformat(),
                'prompt': prompt,
                'visual_theme': visual_theme
            }
        
        # Start background thread
        thread = threading.Thread(
            target=generate_video_task,
            args=(job_id, prompt, visual_theme, output_dir),
            daemon=True
        )
        thread.start()
        
        logger.info(f"📝 Job {job_id} created for prompt: {prompt[:50]}...")
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video generation started'
        }), 202
    
    except Exception as e:
        logger.error(f"❌ Error creating job: {e}")
        return jsonify({'error': f'Failed to create job: {str(e)}'}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job status"""
    job = get_job_status(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify({
        'job_id': job_id,
        'status': job.get('status'),
        'progress': job.get('progress', 0),
        'stage': job.get('message', ''),
        'message': job.get('message', ''),
        'video_ready': job.get('status') == 'completed',
        'video_url': f'/api/download/{job_id}' if job.get('status') == 'completed' else None,
        'error': job.get('message') if job.get('status') == 'failed' else None,
        'updated_at': job.get('updated_at')
    })

@app.route('/api/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download generated video"""
    job = get_job_status(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    video_path = job.get('video_path')
    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(video_path,
                    mimetype='video/mp4',
                    as_attachment=True,
                    download_name=f'generated_{job_id}.mp4')

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    with job_lock:
        job_list = [
            {
                'job_id': j_id,
                'status': job['status'],
                'progress': job.get('progress', 0),
                'created_at': job.get('created_at'),
                'prompt': job.get('prompt', '')[:50]
            }
            for j_id, job in jobs.items()
        ]
    return jsonify({'jobs': job_list})

# =====================================================================
# SOCKET EVENTS
# =====================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to video generation server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to job updates"""
    job_id = data.get('job_id')
    if job_id:
        from flask_socketio import join_room
        join_room(job_id)
        emit('subscribed', {'job_id': job_id})
        logger.info(f"📡 Client subscribed to job {job_id}")

# =====================================================================
# FRONTEND ROUTES (SERVE PWA FILES)
# =====================================================================

@app.route('/')
def serve_index():
    """Serve main PWA index.html"""
    return send_from_directory('VideoGen-PWA', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve JS, CSS, icons, manifest, etc."""
    return send_from_directory('VideoGen-PWA', path)

# =====================================================================
# ERROR HANDLERS
# =====================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    # Initialize config
    if not Config.initialize():
        logger.error("❌ Failed to initialize configuration")
        exit(1)
    
    # Print startup info
    logger.info("=" * 70)
    logger.info("AI VIDEO GENERATION API SERVER")
    logger.info("=" * 70)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Text Model: {Config.HF_MODEL_NAME}")
    logger.info(f"Image Generation: {'Enabled' if Config.USE_IMAGE_GENERATION else 'Disabled'}")
    logger.info(f"Video Resolution: {Config.VIDEO_WIDTH}x{Config.VIDEO_HEIGHT}")
    logger.info("=" * 70)
    
    # Start server
    socketio.run(app, host='0.0.0.0', port=5000,
                debug=True, allow_unsafe_werkzeug=True)
