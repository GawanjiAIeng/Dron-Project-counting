document.addEventListener('DOMContentLoaded', () => {
  // State
  let state = {
    file: null,
    fileType: null,
    results: null,
    currentFrame: 0,
    frames: [],
    activeClasses: new Set(),
    imgsz: 480,
    skip: 5,
    processing: false,
    abortController: null,
    isPlaying: false,
    playInterval: null,
  };

  // DOM refs
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const changeFileBtn = document.getElementById('changeFileBtn');
  const stopBtn = document.getElementById('stopBtn');

  const viewerPlaceholder = document.getElementById('viewerPlaceholder');
  const viewerContent = document.getElementById('viewerContent');
  const viewerImage = document.getElementById('viewerImage');
  const processingOverlay = document.getElementById('processingOverlay');
  const spinnerText = document.getElementById('spinnerText');

  const progressWrapper = document.getElementById('progressWrapper');
  const streamProgressBar = document.getElementById('streamProgressBar');
  const progressText = document.getElementById('progressText');
  const progressPercent = document.getElementById('progressPercent');
  const fpsBadge = document.getElementById('fpsBadge');
  const fpsText = document.getElementById('fpsText');

  const frameCounter = document.getElementById('frameCounter');
  const playPauseBtn = document.getElementById('playPauseBtn');
  const prevFrame = document.getElementById('prevFrame');
  const nextFrame = document.getElementById('nextFrame');

  const scrubberContainer = document.getElementById('scrubberContainer');
  const videoSlider = document.getElementById('videoSlider');

  const chipsContainer = document.getElementById('chipsContainer');
  const clearFilters = document.getElementById('clearFilters');
  const resultsList = document.getElementById('resultsList');
  const detectionCount = document.getElementById('detectionCount');
  const statusBadge = document.getElementById('statusBadge');

  const resGroup = document.getElementById('resGroup');
  const skipGroup = document.getElementById('skipGroup');

  // Init active classes (all)
  document.querySelectorAll('.chip').forEach(ch => state.activeClasses.add(ch.dataset.class));
  updateDetectionCount(0);

  // Upload handlers
  browseBtn.addEventListener('click', () => fileInput.click());
  changeFileBtn.addEventListener('click', () => {
    stopCurrentTask();
    fileInput.click();
  });

  uploadZone.addEventListener('click', (e) => {
    if (e.target === browseBtn || e.target.closest('.file-info')) return;
    fileInput.click();
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  stopBtn.addEventListener('click', () => {
    stopCurrentTask();
    setStatus('Stopped by user', '#ffaa00');
    stopBtn.hidden = true;
    if (state.frames.length > 0) {
      enablePlaybackControls();
    }
  });

  function stopCurrentTask() {
    if (state.abortController) {
      state.abortController.abort();
      state.abortController = null;
    }
    stopPlayback();
    setProcessing(false);
  }

  function handleFile(file) {
    stopCurrentTask();
    const ext = file.name.split('.').pop().toLowerCase();
    const isVideo = ['mp4', 'avi', 'mov', 'webm'].includes(ext);
    state.file = file;
    state.fileType = isVideo ? 'video' : 'image';
    state.currentFrame = 0;
    state.frames = [];
    state.results = null;

    fileName.textContent = file.name;
    uploadZone.hidden = true;
    fileInfo.hidden = false;

    if (isVideo) {
      processVideoStream(file);
    } else {
      processImage(file);
    }
  }

  // --- FAST IMAGE PROCESSING ---
  async function processImage(file) {
    setProcessing(true, 'Detecting objects...');
    setStatus('Processing image...', '#ffcc00');
    fpsBadge.hidden = true;
    progressWrapper.hidden = true;
    scrubberContainer.hidden = true;
    frameCounter.hidden = true;
    playPauseBtn.hidden = true;
    prevFrame.hidden = true;
    nextFrame.hidden = true;
    stopBtn.hidden = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('imgsz', state.imgsz);

    state.abortController = new AbortController();

    try {
      const res = await fetch('/detect', {
        method: 'POST',
        body: formData,
        signal: state.abortController.signal
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      state.results = data;
      viewerImage.src = `data:image/jpeg;base64,${data.image}`;
      viewerPlaceholder.hidden = true;
      viewerContent.hidden = false;

      renderResults(data.detections);
      setStatus(`Detected ${data.detections.length} object(s)`, '#00ff88');
    } catch (err) {
      if (err.name !== 'AbortError') {
        setStatus('Error', '#ff4466');
        resultsList.innerHTML = `<div class="results-empty"><p>Error: ${err.message}</p></div>`;
      }
    } finally {
      setProcessing(false);
      state.abortController = null;
    }
  }

  // --- REAL-TIME LIVE VIDEO STREAMING ---
  async function processVideoStream(file) {
    setProcessing(true, 'Starting live stream...');
    setStatus('Connecting stream...', '#00d4ff');
    
    stopBtn.hidden = false;
    progressWrapper.hidden = false;
    streamProgressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressText.textContent = 'Analyzing video...';
    
    fpsBadge.hidden = false;
    fpsText.textContent = '0 FPS';
    
    viewerPlaceholder.hidden = true;
    viewerContent.hidden = false;
    scrubberContainer.hidden = true;
    frameCounter.hidden = false;
    frameCounter.textContent = '0 frames';
    playPauseBtn.hidden = true;
    prevFrame.hidden = true;
    nextFrame.hidden = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('imgsz', state.imgsz);
    formData.append('skip', state.skip);
    formData.append('max_frames', '300');

    state.abortController = new AbortController();

    try {
      const response = await fetch('/detect_stream', {
        method: 'POST',
        body: formData,
        signal: state.abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      setProcessing(false); // Hide overlay as soon as stream opens

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop(); // keep trailing incomplete chunk

        for (const block of parts) {
          const trimmed = block.trim();
          if (trimmed.startsWith('data: ')) {
            try {
              const payload = JSON.parse(trimmed.slice(6));
              handleStreamPayload(payload);
            } catch (jsonErr) {
              console.warn('Malformed stream JSON:', jsonErr);
            }
          }
        }
      }

      setStatus(`Complete — ${state.frames.length} frames`, '#00ff88');
      stopBtn.hidden = true;
      enablePlaybackControls();
    } catch (err) {
      if (err.name !== 'AbortError') {
        setStatus('Error streaming video', '#ff4466');
        resultsList.innerHTML = `<div class="results-empty"><p>Stream error: ${err.message}</p></div>`;
      }
    } finally {
      setProcessing(false);
      state.abortController = null;
      stopBtn.hidden = true;
      if (state.frames.length > 0) {
        enablePlaybackControls();
      }
    }
  }

  function handleStreamPayload(payload) {
    if (payload.type === 'frame') {
      const frameData = {
        frame: payload.frame,
        detections: payload.detections,
        image: payload.image
      };
      state.frames.push(frameData);
      state.currentFrame = state.frames.length - 1;

      // Update live image immediately
      viewerImage.src = `data:image/jpeg;base64,${payload.image}`;

      // Update FPS badge
      if (payload.current_fps) {
        fpsText.textContent = `${payload.current_fps} FPS`;
      }

      // Update progress bar
      if (payload.progress !== undefined) {
        streamProgressBar.style.width = `${payload.progress}%`;
        progressPercent.textContent = `${payload.progress}%`;
        progressText.textContent = `Streaming frame ${payload.frame} of ~${payload.estimated_total}`;
      }

      frameCounter.textContent = `Frame ${payload.frame}`;
      renderResults(payload.detections);
      setStatus(`Live: ${payload.detections.length} objects`, '#00d4ff');
    } else if (payload.type === 'complete') {
      streamProgressBar.style.width = '100%';
      progressPercent.textContent = '100%';
      progressText.textContent = `Finished ${payload.total_processed} frames in ${payload.elapsed_sec}s`;
      setStatus(`Done — ${payload.total_processed} frames`, '#00ff88');
      enablePlaybackControls();
    }
  }

  function enablePlaybackControls() {
    if (!state.frames.length) return;
    scrubberContainer.hidden = false;
    videoSlider.max = state.frames.length - 1;
    videoSlider.value = state.currentFrame;
    frameCounter.hidden = false;
    frameCounter.textContent = `${state.currentFrame + 1} / ${state.frames.length}`;
    playPauseBtn.hidden = false;
    playPauseBtn.textContent = 'Play';
    prevFrame.hidden = false;
    nextFrame.hidden = false;
    updateNavButtons();
  }

  function showFrame(index) {
    if (!state.frames.length) return;
    if (index < 0) index = 0;
    if (index >= state.frames.length) index = state.frames.length - 1;

    state.currentFrame = index;
    const frame = state.frames[index];
    if (!frame) return;

    viewerImage.src = `data:image/jpeg;base64,${frame.image}`;
    videoSlider.value = index;
    frameCounter.textContent = `${index + 1} / ${state.frames.length}`;
    renderResults(frame.detections);
    updateNavButtons();
  }

  function updateNavButtons() {
    prevFrame.disabled = state.currentFrame === 0;
    nextFrame.disabled = state.currentFrame === state.frames.length - 1;
  }

  // Scrubber slider interaction
  videoSlider.addEventListener('input', (e) => {
    stopPlayback();
    showFrame(parseInt(e.target.value));
  });

  // Next / Prev button interaction
  prevFrame.addEventListener('click', () => {
    stopPlayback();
    showFrame(state.currentFrame - 1);
  });
  nextFrame.addEventListener('click', () => {
    stopPlayback();
    showFrame(state.currentFrame + 1);
  });

  // Play / Pause interaction
  playPauseBtn.addEventListener('click', () => {
    if (state.isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  function startPlayback() {
    if (!state.frames.length) return;
    state.isPlaying = true;
    playPauseBtn.textContent = 'Pause';
    state.playInterval = setInterval(() => {
      let next = state.currentFrame + 1;
      if (next >= state.frames.length) next = 0;
      showFrame(next);
    }, 150);
  }

  function stopPlayback() {
    state.isPlaying = false;
    playPauseBtn.textContent = 'Play';
    if (state.playInterval) {
      clearInterval(state.playInterval);
      state.playInterval = null;
    }
  }

  // Object category filters
  chipsContainer.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const cls = chip.dataset.class;
    if (state.activeClasses.has(cls)) {
      state.activeClasses.delete(cls);
      chip.classList.remove('active');
    } else {
      state.activeClasses.add(cls);
      chip.classList.add('active');
    }
    applyFilters();
  });

  clearFilters.addEventListener('click', () => {
    document.querySelectorAll('.chip').forEach(ch => {
      ch.classList.add('active');
      state.activeClasses.add(ch.dataset.class);
    });
    applyFilters();
  });

  function applyFilters() {
    if (state.results && state.results.type === 'image') {
      renderResults(state.results.detections);
    } else if (state.frames.length) {
      const frame = state.frames[state.currentFrame];
      if (frame) renderResults(frame.detections);
    }
  }

  function renderResults(detections) {
    const filtered = (detections || []).filter(d => state.activeClasses.has(d.class));
    updateDetectionCount(filtered.length);

    if (!filtered.length) {
      resultsList.innerHTML = `<div class="results-empty"><p>No objects detected</p></div>`;
      return;
    }

    resultsList.innerHTML = '';
    const fragment = document.createDocumentFragment();

    filtered.forEach((d, i) => {
      const item = document.createElement('div');
      item.className = 'detection-item';
      const pct = (d.confidence * 100).toFixed(1);

      item.innerHTML = `
        <div class="detection-top">
          <span class="detection-class">${d.class}</span>
          <span class="detection-conf">${pct}%</span>
        </div>
        <div class="conf-bar">
          <div class="conf-fill" style="width: ${pct}%"></div>
        </div>
      `;
      fragment.appendChild(item);
    });

    resultsList.appendChild(fragment);
  }

  function updateDetectionCount(n) {
    detectionCount.textContent = n ?? '0';
  }

  // Resolution setting toggle
  resGroup.addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-setting');
    if (!btn) return;
    resGroup.querySelectorAll('.btn-setting').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.imgsz = parseInt(btn.dataset.value);
  });

  // Skip rate toggle
  skipGroup.addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-setting');
    if (!btn) return;
    skipGroup.querySelectorAll('.btn-setting').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.skip = parseInt(btn.dataset.value);
  });

  // Helpers
  function setProcessing(val, msg) {
    state.processing = val;
    processingOverlay.hidden = !val;
    if (msg && spinnerText) spinnerText.textContent = msg;
  }

  function setStatus(text, color) {
    statusBadge.innerHTML = `
      <span class="pulse-dot" style="background: ${color || '#00ff88'}; box-shadow: 0 0 0 0 ${color || '#00ff88'}40;"></span>
      ${text}
    `;
  }
});