/**
 * AI Video Generator - Clean Same-Origin Version
 * Uses relative API paths (no IP hardcoding, no auto-detection)
 */

let currentJobId = null;
let pollingInterval = null;

// DOM Elements
const elements = {
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
    productDescription: document.getElementById('product-description'),
    themeSelect: document.getElementById('theme-select'),
    generateBtn: document.getElementById('generate-btn'),
    progressSection: document.getElementById('progress-section'),
    progressBar: document.getElementById('progress-bar'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressStage: document.getElementById('progress-stage'),
    resultSection: document.getElementById('result-section'),
    resultVideo: document.getElementById('result-video'),
    errorSection: document.getElementById('error-section'),
    errorMessage: document.getElementById('error-message')
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 App initialized');
    await checkHealth();
    setupEventListeners();
});

// -----------------------------
// Health Check
// -----------------------------

async function checkHealth() {
    updateConnectionStatus('connecting', 'Connecting...');

    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error();

        const data = await response.json();
        updateConnectionStatus('connected', `Connected to ${data.service}`);
    } catch (error) {
        updateConnectionStatus('disconnected', 'Server Offline');
    }
}

function updateConnectionStatus(status, text) {
    if (elements.statusIndicator) {
        elements.statusIndicator.className = `status-dot status-${status}`;
    }
    if (elements.statusText) {
        elements.statusText.textContent = text;
    }
}

// -----------------------------
// Generate Video
// -----------------------------

async function generateVideo() {
    const prompt = elements.productDescription.value.trim();
    const theme = elements.themeSelect.value;

    if (!prompt) {
        alert('Please enter a product description');
        return;
    }

    elements.generateBtn.disabled = true;
    elements.generateBtn.textContent = 'Generating...';
    showSection(elements.progressSection);

    try {
        const response = await fetch('/api/generate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, theme })
        });

        if (!response.ok) throw new Error();

        const data = await response.json();
        currentJobId = data.job_id;
        startPolling(currentJobId);

    } catch (error) {
        showError('Failed to start generation.');
    }
}

// -----------------------------
// Polling
// -----------------------------

function startPolling(jobId) {
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${jobId}`);
            if (!response.ok) throw new Error();

            const status = await response.json();
            updateProgress(status.progress || 0, status.message || '');

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                handleCompletion(status.video_url);
            }

            if (status.status === 'failed') {
                clearInterval(pollingInterval);
                showError(status.error || 'Generation failed');
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

// -----------------------------
// UI Helpers
// -----------------------------

function updateProgress(progress, stage) {
    elements.progressBar.style.width = `${progress}%`;
    elements.progressPercentage.textContent = `${Math.round(progress)}%`;
    elements.progressStage.textContent = stage;
}

function handleCompletion(videoUrl) {
    hideSection(elements.progressSection);
    showSection(elements.resultSection);
    elements.resultVideo.src = videoUrl;
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Video';
}

function showError(message) {
    hideSection(elements.progressSection);
    showSection(elements.errorSection);
    elements.errorMessage.textContent = message;
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Video';
}

function showSection(el) {
    if (el) el.classList.remove('hidden');
}

function hideSection(el) {
    if (el) el.classList.add('hidden');
}

function setupEventListeners() {
    if (elements.generateBtn) {
        elements.generateBtn.addEventListener('click', generateVideo);
    }
}
