/**
 * AI Video Generator PWA - SMART NETWORK DETECTION
 * Automatically detects and connects to the correct server
 */

// Configuration - Multiple possible server URLs
const POSSIBLE_SERVERS = [
    'http://192.168.20.4:5000',  
    'http://10.10.48.96:5000',   
    'http://localhost:5000',     
];

let API_BASE_URL = localStorage.getItem('api_base_url') || null;
let currentJobId = null;
let pollingInterval = null;
let socket = null;

// DOM Elements
const elements = {
    serverUrl: document.getElementById('server-url'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
    productDescription: document.getElementById('product-description'),
    themeSelect: document.getElementById('theme-select'),
    generateBtn: document.getElementById('generate-btn'),
    progressSection: document.getElementById('progress-section'),
    progressBar: document.getElementById('progress-bar'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressStage: document.getElementById('progress-stage'),
    cancelBtn: document.getElementById('cancel-btn'),
    resultSection: document.getElementById('result-section'),
    resultVideo: document.getElementById('result-video'),
    downloadBtn: document.getElementById('download-btn'),
    shareBtn: document.getElementById('share-btn'),
    newVideoBtn: document.getElementById('new-video-btn'),
    errorSection: document.getElementById('error-section'),
    errorMessage: document.getElementById('error-message'),
    retryBtn: document.getElementById('retry-btn')
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 App initialized');
    
    // Auto-detect server if not manually set
    if (!API_BASE_URL) {
        console.log('🔍 No saved server, auto-detecting...');
        await autoDetectServer();
    } else {
        console.log('📡 Using saved server:', API_BASE_URL);
        if (elements.serverUrl) {
            elements.serverUrl.value = API_BASE_URL;
        }
        // Test if saved server still works
        const connected = await testConnection();
        if (!connected) {
            console.log('⚠️ Saved server not responding, trying auto-detect...');
            await autoDetectServer();
        }
    }
    
    // Setup event listeners
    setupEventListeners();
});

/**
 * Auto-detect which server is available
 */
async function autoDetectServer() {
    updateConnectionStatus('connecting', 'Detecting server...');
    
    for (const serverUrl of POSSIBLE_SERVERS) {
        console.log(`🔍 Trying ${serverUrl}...`);
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
            
            const response = await fetch(`${serverUrl}/api/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                mode: 'cors',
                cache: 'no-cache',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                console.log(`✅ Found server at ${serverUrl}`);
                API_BASE_URL = serverUrl;
                localStorage.setItem('api_base_url', serverUrl);
                
                if (elements.serverUrl) {
                    elements.serverUrl.value = serverUrl;
                }
                
                updateConnectionStatus('connected', `Connected to ${data.service || 'Server'}`);
                return true;
            }
        } catch (error) {
            console.log(`❌ ${serverUrl} not available`);
            // Continue to next server
        }
    }
    
    // No server found
    console.error('❌ No server detected on any known network');
    updateConnectionStatus('disconnected', 'No server found');
    alert('Cannot find server. Please:\n1. Make sure the server is running\n2. Check both devices are on the same WiFi\n3. Or manually enter the server URL');
    return false;
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    if (elements.saveSettingsBtn) {
        elements.saveSettingsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            saveSettings();
        });
    }

    if (elements.generateBtn) {
        elements.generateBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            generateVideo();
        });
    }

    if (elements.cancelBtn) {
        elements.cancelBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            cancelGeneration();
        });
    }

    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            downloadVideo();
        });
    }

    if (elements.shareBtn) {
        elements.shareBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            shareVideo();
        });
    }

    if (elements.newVideoBtn) {
        elements.newVideoBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            resetForm();
        });
    }

    if (elements.retryBtn) {
        elements.retryBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            generateVideo();
        });
    }

    document.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
        }
    });
}

/**
 * Save server settings (Manual override)
 */
function saveSettings() {
    const newUrl = elements.serverUrl.value.trim();

    if (!newUrl) {
        alert('Please enter a server URL');
        return;
    }

    if (!newUrl.startsWith('http://') && !newUrl.startsWith('https://')) {
        alert('URL must start with http:// or https://');
        return;
    }

    API_BASE_URL = newUrl;
    localStorage.setItem('api_base_url', newUrl);

    console.log('✅ Settings saved:', newUrl);
    alert('Settings saved! Testing connection...');

    testConnection();
}

/**
 * Test connection to current server
 */
async function testConnection() {
    if (!API_BASE_URL) {
        console.log('⚠️ No server URL set');
        return false;
    }

    console.log('🔍 Testing connection to:', API_BASE_URL);
    updateConnectionStatus('connecting', 'Connecting...');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const response = await fetch(`${API_BASE_URL}/api/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            mode: 'cors',
            cache: 'no-cache',
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        if (response.ok) {
            const data = await response.json();
            console.log('✅ Connected to server:', data);
            updateConnectionStatus('connected', `Connected to ${data.service || 'Server'}`);
            return true;
        } else {
            throw new Error(`Server returned ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Connection failed:', error);
        updateConnectionStatus('disconnected', 'Cannot connect');
        return false;
    }
}

/**
 * Update connection status UI
 */
function updateConnectionStatus(status, text) {
    if (elements.statusIndicator) {
        elements.statusIndicator.className = `status-dot status-${status}`;
    }
    if (elements.statusText) {
        elements.statusText.textContent = text;
    }
}

/**
 * Generate video
 */
async function generateVideo() {
    console.log('🎬 Generate video clicked');

    const prompt = elements.productDescription.value.trim();
    const theme = elements.themeSelect.value;

    if (!prompt) {
        alert('Please enter a product description');
        return;
    }

    // Try to reconnect if needed
    let connected = await testConnection();
    if (!connected) {
        console.log('🔄 Attempting to reconnect...');
        connected = await autoDetectServer();
        if (!connected) {
            alert('Cannot connect to server. Please check your server URL and ensure both devices are on the same WiFi.');
            return;
        }
    }

    hideAllSections();
    showSection(elements.progressSection);
    updateProgress(0, 'Initializing...');
    elements.generateBtn.disabled = true;
    elements.generateBtn.textContent = 'Generating...';

    try {
        const response = await fetch(`${API_BASE_URL}/api/generate-video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            mode: 'cors',
            cache: 'no-cache',
            body: JSON.stringify({ prompt: prompt, theme: theme })
        });

        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const data = await response.json();
        console.log('✅ Job created:', data);
        currentJobId = data.job_id;

        startPolling(currentJobId);
    } catch (error) {
        console.error('❌ Generation failed:', error);
        showError('Failed to start generation: ' + error.message);
        elements.generateBtn.disabled = false;
        elements.generateBtn.textContent = 'Generate Video';
    }
}

function startPolling(jobId) {
    console.log('🔄 Starting status polling for job:', jobId);

    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/status/${jobId}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                mode: 'cors',
                cache: 'no-cache'
            });

            if (!response.ok) throw new Error(`Status check failed: ${response.status}`);

            const status = await response.json();
            console.log('📊 Status update:', status);

            updateProgress(status.progress || 0, status.message || 'Processing...');

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                handleCompletion(status.video_url);
            } else if (status.status === 'failed') {
                clearInterval(pollingInterval);
                showError(status.error || 'Generation failed');
            }
        } catch (error) {
            console.error('❌ Status check error:', error);
        }
    }, 2000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

function updateProgress(progress, stage) {
    if (elements.progressBar) elements.progressBar.style.width = `${progress}%`;
    if (elements.progressPercentage) elements.progressPercentage.textContent = `${Math.round(progress)}%`;
    if (elements.progressStage) elements.progressStage.textContent = stage;
    updateStepIndicators(progress);
}

function updateStepIndicators(progress) {
    const steps = ['storyline', 'images', 'audio', 'video'];
    const thresholds = [25, 50, 75, 100];
    steps.forEach((step, index) => {
        const element = document.getElementById(`step-${step}`);
        if (element) {
            if (progress >= thresholds[index]) element.classList.add('active');
            else element.classList.remove('active');
        }
    });
}

function handleCompletion(videoUrl) {
    console.log('✅ Generation complete! Video URL:', videoUrl);
    hideSection(elements.progressSection);
    showSection(elements.resultSection);
    if (elements.resultVideo && videoUrl) elements.resultVideo.src = `${API_BASE_URL}${videoUrl}`;
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Video';
}

function showError(message) {
    console.error('❌ Error:', message);
    hideAllSections();
    showSection(elements.errorSection);
    if (elements.errorMessage) elements.errorMessage.textContent = message;
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Video';
}

function cancelGeneration() {
    console.log('🛑 Canceling generation');
    stopPolling();
    currentJobId = null;
    resetForm();
}

function downloadVideo() {
    if (!currentJobId) return;
    const downloadUrl = `${API_BASE_URL}/api/download/${currentJobId}`;
    console.log('📥 Downloading video from:', downloadUrl);
    window.open(downloadUrl, '_blank');
}

async function shareVideo() {
    if (!elements.resultVideo || !elements.resultVideo.src) return;
    const videoUrl = elements.resultVideo.src;

    if (navigator.share) {
        try {
            await navigator.share({
                title: 'AI Generated Video',
                text: 'Check out this video I created!',
                url: videoUrl
            });
            console.log('✅ Video shared successfully');
        } catch (error) {
            console.log('❌ Sharing failed:', error);
        }
    } else {
        try {
            await navigator.clipboard.writeText(videoUrl);
            alert('Video URL copied to clipboard!');
        } catch (error) {
            alert('Sharing not supported on this device');
        }
    }
}

function resetForm() {
    console.log('🔄 Resetting form');
    stopPolling();
    currentJobId = null;
    hideAllSections();
    updateProgress(0, 'Initializing...');
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Video';
    if (elements.resultVideo) elements.resultVideo.src = '';
}

function showSection(element) {
    if (element) element.classList.remove('hidden');
}

function hideSection(element) {
    if (element) element.classList.add('hidden');
}

function hideAllSections() {
    hideSection(elements.progressSection);
    hideSection(elements.resultSection);
    hideSection(elements.errorSection);
}

// Auto-reconnect when network changes
document.addEventListener('visibilitychange', async () => {
    if (!document.hidden) {
        if (currentJobId && !pollingInterval) {
            startPolling(currentJobId);
        } else {
            // Check connection when app becomes visible
            const connected = await testConnection();
            if (!connected) {
                await autoDetectServer();
            }
        }
    }
});

window.addEventListener('online', async () => {
    console.log('🌐 Back online, detecting server...');
    updateConnectionStatus('connecting', 'Reconnecting...');
    await autoDetectServer();
});

window.addEventListener('offline', () => {
    console.log('📴 Offline');
    updateConnectionStatus('disconnected', 'Offline');
});

console.log('✅ App.js loaded with smart network detection');