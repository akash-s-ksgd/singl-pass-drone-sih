/**
 * Drone 3D Reconstruction — Main Application
 * Handles navigation, upload workflow, processing, 3D viewer, and report display.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = '/api';

// ─── State ──────────────────────────────────────────────────
const state = {
    projectId: null,
    jobId: null,
    pollingInterval: null,
    viewer: null,
    pointCloud: null,
};

// ─── Navigation ─────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const page = btn.dataset.page;
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(`page-${page}`).classList.add('active');
        btn.classList.add('active');
        if (page === 'viewer' && !state.viewer) initViewer();
    });
});

// ─── Health Check ───────────────────────────────────────────
async function checkHealth() {
    const el = document.getElementById('system-status');
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        const dot = el.querySelector('.status-dot');
        dot.classList.add('online');
        const gpuText = data.gpu_available ? `GPU: ${data.gpus && data.gpus.length > 0 ? data.gpus[0].name : 'Yes'}` : 'CPU Mode';
        const colmapText = data.colmap_available ? '• COLMAP ✓' : '• COLMAP ✗';
        el.querySelector('span').textContent = `${gpuText} ${colmapText}`;

        // Populate GPU select
        if (data.gpu_available && data.gpus && data.gpus.length > 0) {
            const select = document.getElementById('cfg-gpu-device');
            select.innerHTML = '';
            data.gpus.forEach(gpu => {
                const opt = document.createElement('option');
                opt.value = gpu.id;
                opt.textContent = `${gpu.id} - ${gpu.name}`;
                opt.style.background = '#1a2236';
                select.appendChild(opt);
            });
            document.getElementById('gpu-select-group').style.display = document.getElementById('cfg-gpu').checked ? 'block' : 'none';
        }
    } catch {
        el.querySelector('span').textContent = 'API Offline';
    }
}
checkHealth();

// Add listener for GPU toggle
document.getElementById('cfg-gpu').addEventListener('change', (e) => {
    const selectGroup = document.getElementById('gpu-select-group');
    if (e.target.checked && document.getElementById('cfg-gpu-device').options.length > 0 && document.getElementById('cfg-gpu-device').options[0].value !== '') {
        selectGroup.style.display = 'block';
    } else {
        selectGroup.style.display = 'none';
    }
});

// ─── Upload Workflow ────────────────────────────────────────

// Step 1: Create Project
document.getElementById('btn-create-project').addEventListener('click', async () => {
    const name = document.getElementById('project-name').value.trim();
    if (!name) { document.getElementById('project-name').focus(); return; }

    try {
        const res = await fetch(`${API}/projects`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: document.getElementById('project-desc').value }),
        });
        const data = await res.json();
        state.projectId = data.id;

        document.getElementById('card-project').style.opacity = '0.5';
        document.getElementById('card-upload').classList.remove('disabled');
        showToast(`Project "${name}" created`);
    } catch (e) {
        showToast('Failed to create project', 'error');
    }
});

// Step 2: Upload Video
const uploadZone = document.getElementById('upload-zone');
const videoInput = document.getElementById('video-input');

document.getElementById('btn-browse').addEventListener('click', (e) => {
    e.stopPropagation();
    videoInput.click();
});
uploadZone.addEventListener('click', () => videoInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
});
videoInput.addEventListener('change', () => {
    if (videoInput.files.length) uploadVideo(videoInput.files[0]);
});

async function uploadVideo(file) {
    if (!state.projectId) return;

    const progressEl = document.getElementById('upload-progress');
    const fillEl = document.getElementById('upload-fill');
    const textEl = document.getElementById('upload-text');
    const infoEl = document.getElementById('video-info');

    progressEl.classList.remove('hidden');
    uploadZone.classList.add('hidden');

    const formData = new FormData();
    formData.append('video', file);

    try {
        // Simulate progress (XHR for real progress)
        let progress = 0;
        const interval = setInterval(() => {
            progress = Math.min(progress + 5, 90);
            fillEl.style.width = `${progress}%`;
            textEl.textContent = `Uploading... ${progress}%`;
        }, 200);

        const res = await fetch(`${API}/projects/${state.projectId}/upload`, {
            method: 'POST',
            body: formData,
        });
        clearInterval(interval);
        fillEl.style.width = '100%';
        textEl.textContent = 'Upload complete!';

        const data = await res.json();
        infoEl.innerHTML = `
            <span>File:</span> <strong>${data.filename}</strong>
            <span>Size:</span> <strong>${data.file_size_mb} MB</strong>
            <span>Duration:</span> <strong>${data.video_duration_s}s</strong>
            <span>Resolution:</span> <strong>${data.resolution}</strong>
        `;
        infoEl.classList.remove('hidden');

        document.getElementById('card-upload').style.opacity = '0.6';
        document.getElementById('card-config').classList.remove('disabled');
        showToast('Video uploaded successfully');
    } catch (e) {
        textEl.textContent = 'Upload failed!';
        showToast('Upload failed', 'error');
    }
}

// Step 3: Start Processing
document.getElementById('btn-start-processing').addEventListener('click', async () => {
    if (!state.projectId) return;

    const config = {
        fps: parseFloat(document.getElementById('cfg-fps').value),
        max_frames: parseInt(document.getElementById('cfg-max-frames').value),
        min_sharpness: parseFloat(document.getElementById('cfg-sharpness').value),
        min_feature_count: parseInt(document.getElementById('cfg-features').value),
        use_gpu: document.getElementById('cfg-gpu').checked,
        gpu_device: document.getElementById('cfg-gpu-device').value || null,
        enable_depth_fusion: document.getElementById('cfg-depth-fusion').checked,
        enable_dynamic_removal: document.getElementById('cfg-dynamic-removal').checked,
    };

    try {
        const res = await fetch(`${API}/projects/${state.projectId}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await res.json();
        state.jobId = data.job_id;

        // Switch to dashboard
        document.getElementById('nav-dashboard').click();
        startPolling();
        showToast('Processing started!');
    } catch (e) {
        showToast('Failed to start processing', 'error');
    }
});

// ─── Polling ────────────────────────────────────────────────

const stageOrder = [
    'extracting_frames', 'selecting_frames', 'estimating_poses',
    'estimating_depth', 'reconstructing', 'generating_mesh', 'completed'
];

function startPolling() {
    if (state.pollingInterval) clearInterval(state.pollingInterval);
    state.pollingInterval = setInterval(pollStatus, 1500);
    pollStatus();
}

async function pollStatus() {
    if (!state.jobId) return;
    try {
        const res = await fetch(`${API}/jobs/${state.jobId}`);
        const job = await res.json();
        updateDashboard(job);

        if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(state.pollingInterval);
            if (job.status === 'completed') {
                showToast('Reconstruction complete! 🎉');
                loadReport();
            } else {
                showToast(`Failed: ${job.error || 'Unknown error'}`, 'error');
            }
        }
    } catch { /* retry next interval */ }
}

function updateDashboard(job) {
    // Update pipeline stages
    const currentIdx = stageOrder.indexOf(job.status);
    document.querySelectorAll('.stage').forEach(stage => {
        const stageStatus = stage.dataset.stage;
        const idx = stageOrder.indexOf(stageStatus);
        stage.classList.remove('active', 'completed');
        if (idx < currentIdx) stage.classList.add('completed');
        else if (idx === currentIdx) stage.classList.add('active');
    });

    // Progress bar
    document.getElementById('pipeline-fill').style.width = `${(job.progress * 100).toFixed(0)}%`;
    document.getElementById('pipeline-message').textContent = job.message || job.current_step;

    // Stats
    if (job.frames_extracted) document.getElementById('stat-frames').textContent = job.frames_extracted.toLocaleString();
    if (job.frames_selected) document.getElementById('stat-selected').textContent = job.frames_selected.toLocaleString();
    if (job.points_reconstructed) document.getElementById('stat-points').textContent = job.points_reconstructed.toLocaleString();
}

// ─── Report ─────────────────────────────────────────────────

async function loadReport() {
    if (!state.jobId) return;
    try {
        const res = await fetch(`${API}/jobs/${state.jobId}/report`);
        const report = await res.json();
        renderReport(report);
    } catch { /* not ready yet */ }
}

function renderReport(report) {
    const el = document.getElementById('report-content');

    // Confidence distribution
    const confDist = report.confidence_distribution || [];
    const high = confDist.find(c => c.level === 'high') || { percentage: 0 };
    const med = confDist.find(c => c.level === 'medium') || { percentage: 0 };
    const low = confDist.find(c => c.level === 'low') || { percentage: 0 };

    el.innerHTML = `
        <!-- Overview -->
        <div class="card glass">
            <h2>📊 Overview</h2>
            <div class="metric-row"><span class="metric-label">Processing Time</span><span class="metric-value">${report.processing_time_seconds}s</span></div>
            <div class="metric-row"><span class="metric-label">Frames Extracted</span><span class="metric-value">${report.frames_extracted}</span></div>
            <div class="metric-row"><span class="metric-label">Frames Selected</span><span class="metric-value">${report.frames_selected}</span></div>
            <div class="metric-row"><span class="metric-label">Camera Poses</span><span class="metric-value">${report.camera_poses_recovered}</span></div>
            <div class="metric-row"><span class="metric-label">Dense Points</span><span class="metric-value">${report.dense_points?.toLocaleString()}</span></div>
            <div class="metric-row"><span class="metric-label">Mesh Vertices</span><span class="metric-value">${report.mesh_vertices?.toLocaleString()}</span></div>
            <div class="metric-row"><span class="metric-label">Mesh Faces</span><span class="metric-value">${report.mesh_faces?.toLocaleString()}</span></div>
        </div>

        <!-- Confidence -->
        <div class="card glass">
            <h2>🎯 Confidence Distribution</h2>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
                Shows how much of the model is directly observed vs. AI-inferred
            </p>
            <div class="confidence-bar-container">
                <div class="confidence-bar">
                    <div class="confidence-segment confidence-high" style="width:${high.percentage}%">${high.percentage > 5 ? high.percentage + '%' : ''}</div>
                    <div class="confidence-segment confidence-medium" style="width:${med.percentage}%">${med.percentage > 5 ? med.percentage + '%' : ''}</div>
                    <div class="confidence-segment confidence-low" style="width:${low.percentage}%">${low.percentage > 5 ? low.percentage + '%' : ''}</div>
                </div>
                <div class="confidence-legend">
                    <div class="legend-item"><div class="legend-dot" style="background:var(--success)"></div>High (Multi-view verified)</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--warning)"></div>Medium (Limited views)</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--danger)"></div>Low (AI-inferred)</div>
                </div>
            </div>
        </div>

        <!-- Metrics -->
        <div class="card glass">
            <h2>📐 Quality Metrics</h2>
            ${Object.entries(report.metrics || {}).map(([k, v]) => `
                <div class="metric-row">
                    <span class="metric-label">${k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    <span class="metric-value">${typeof v === 'number' ? v.toLocaleString() : v}</span>
                </div>
            `).join('')}
        </div>

        <!-- Downloads -->
        <div class="card glass">
            <h2>📁 Output Files</h2>
            ${Object.entries(report.output_files || {}).map(([fmt, path]) => `
                <div class="metric-row">
                    <span class="metric-label">${fmt.toUpperCase()}</span>
                    <a href="${API}/projects/${state.projectId}/models/${path.split('/').pop()}" class="btn btn-outline" style="padding: 0.3rem 0.8rem; font-size: 0.8rem;">Download</a>
                </div>
            `).join('')}
            <button class="btn btn-primary" style="margin-top: 1rem; width: 100%; justify-content: center;" onclick="document.getElementById('nav-viewer').click()">
                Open in 3D Viewer →
            </button>
        </div>
    `;
}

// ─── 3D Viewer ──────────────────────────────────────────────

function initViewer() {
    const canvas = document.getElementById('viewer-canvas');
    const container = canvas.parentElement;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a14);
    scene.fog = new THREE.FogExp2(0x060a14, 0.002);

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(10, 10, 10);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Grid helper
    const grid = new THREE.GridHelper(50, 50, 0x1a2236, 0x111827);
    scene.add(grid);

    // Axes
    const axes = new THREE.AxesHelper(5);
    scene.add(axes);

    // Ambient light
    scene.add(new THREE.AmbientLight(0x404060, 2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    state.viewer = { scene, camera, renderer, controls };

    // Toolbar buttons
    document.getElementById('btn-reset-view').addEventListener('click', () => {
        camera.position.set(10, 10, 10);
        controls.target.set(0, 0, 0);
    });

    document.getElementById('btn-load-pointcloud').addEventListener('click', loadPointCloud);
}

async function loadPointCloud() {
    if (!state.projectId || !state.viewer) return;

    try {
        const res = await fetch(`${API}/projects/${state.projectId}/pointcloud`);
        const data = await res.json();

        if (!data.points || data.points.length === 0) {
            showToast('No point cloud data available', 'error');
            return;
        }

        // Remove old point cloud
        if (state.pointCloud) {
            state.viewer.scene.remove(state.pointCloud);
            state.pointCloud.geometry.dispose();
            state.pointCloud.material.dispose();
        }

        const positions = new Float32Array(data.points.flat());
        const colors = data.colors?.length
            ? new Float32Array(data.colors.flat())
            : new Float32Array(positions.length).fill(0.6);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            sizeAttenuation: true,
        });

        const pointCloud = new THREE.Points(geometry, material);
        state.viewer.scene.add(pointCloud);
        state.pointCloud = pointCloud;

        // Center camera on point cloud
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        state.viewer.controls.target.copy(center);
        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        state.viewer.camera.position.set(
            center.x + maxDim, center.y + maxDim * 0.8, center.z + maxDim
        );

        document.getElementById('viewer-point-count').textContent =
            `${(data.points.length).toLocaleString()} points loaded`;
        showToast(`Loaded ${data.points.length.toLocaleString()} points`);

    } catch (e) {
        showToast('Failed to load point cloud', 'error');
    }
}

// ─── Toast Notifications ────────────────────────────────────

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999;
        padding: 0.75rem 1.25rem; border-radius: 10px;
        font-family: var(--font); font-size: 0.88rem; font-weight: 500;
        color: white; animation: fadeIn 0.3s ease;
        backdrop-filter: blur(12px);
        ${type === 'error'
            ? 'background: rgba(239,68,68,0.85); border: 1px solid rgba(239,68,68,0.5);'
            : 'background: rgba(34,197,94,0.85); border: 1px solid rgba(34,197,94,0.5);'
        }
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ─── Expose for inline handlers ─────────────────────────────
window.showToast = showToast;
