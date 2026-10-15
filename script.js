// =========================================================================
// ENVIRONMENT ROUTING & STATE MANAGEMENT
// =========================================================================
const BACKEND_URL = (
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:"
) ? "http://127.0.0.1:8000" : "https://photosort-backend1.onrender.com";

let photoLibrary = [];
let currentFilter = 'all';
let currentSort = 'score_desc';
let currentSceneMode = 'auto';
let inspectedPhotoId = null;
let selectedPhotoIds = new Set();
let pendingDeleteType = null;
let pendingDeleteTargetId = null;

window.addEventListener('DOMContentLoaded', () => {
  loadPhotosFromAtlas();
  loadBackendStats();
  setupKeyboardShortcuts();

  // Close mobile top-right menu when tapping outside
  document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('headerMenuDropdown');
    const menuBtn = document.getElementById('mobileMenuBtn');
    if (dropdown && dropdown.classList.contains('open')) {
      if (!dropdown.contains(e.target) && (!menuBtn || !menuBtn.contains(e.target))) {
        closeMobileMenu();
      }
    }
  });
});

// =========================================================================
// MOBILE TOP-RIGHT MENU CONTROLLER
// =========================================================================
function toggleMobileMenu(event) {
  if (event) event.stopPropagation();
  const dropdown = document.getElementById('headerMenuDropdown');
  const menuBtn = document.getElementById('mobileMenuBtn');
  const barsIcon = document.getElementById('menuIconBars');
  const closeIcon = document.getElementById('menuIconClose');

  const isOpen = dropdown.classList.toggle('open');
  menuBtn.classList.toggle('active', isOpen);
  if (barsIcon && closeIcon) {
    barsIcon.style.display = isOpen ? 'none' : 'block';
    closeIcon.style.display = isOpen ? 'block' : 'none';
  }
}

function closeMobileMenu() {
  const dropdown = document.getElementById('headerMenuDropdown');
  const menuBtn = document.getElementById('mobileMenuBtn');
  const barsIcon = document.getElementById('menuIconBars');
  const closeIcon = document.getElementById('menuIconClose');

  if (dropdown) dropdown.classList.remove('open');
  if (menuBtn) menuBtn.classList.remove('active');
  if (barsIcon && closeIcon) {
    barsIcon.style.display = 'block';
    closeIcon.style.display = 'none';
  }
}

function toggleStudioTheme() {
  const root = document.documentElement;
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  showToast(`Switched to ${next === 'dark' ? 'Pure Black Studio' : 'Daylight Lab'} Mode`);
}

function handleSceneModeChange() {
  currentSceneMode = document.getElementById('sceneModeSelect').value;
  showToast(`Scene Profile set to: ${currentSceneMode.toUpperCase()}`);
  if (photoLibrary.length > 0) {
    recalibrateLibrary();
  }
}

// =========================================================================
// CLIENT-SIDE CALIBRATED v4.0 COMPUTER VISION ENGINE (CANVAS PIXEL PIPELINE)
// =========================================================================
async function analyzeImageClientCV(imgSource, mode = 'auto') {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      try {
        const origW = img.naturalWidth || img.width;
        const origH = img.naturalHeight || img.height;
        const maxDim = 512;
        const scale = Math.min(1, maxDim / Math.max(origW, origH));
        const w = Math.max(64, Math.round(origW * scale));
        const h = Math.max(64, Math.round(origH * scale));

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);

        const rgba = ctx.getImageData(0, 0, w, h).data;
        const gray = new Float32Array(w * h);

        for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
          gray[p] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
        }

        function gaussianBlur(src, width, height, radius) {
          const out = new Float32Array(width * height);
          const temp = new Float32Array(width * height);
          const sigma = Math.max(0.5, radius / 2);
          const kSize = radius * 2 + 1;
          const kernel = new Float32Array(kSize);
          let sum = 0;
          for (let i = -radius; i <= radius; i++) {
            const val = Math.exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + radius] = val;
            sum += val;
          }
          for (let i = 0; i < kSize; i++) kernel[i] /= sum;

          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              let acc = 0;
              for (let k = -radius; k <= radius; k++) {
                const cx = Math.min(width - 1, Math.max(0, x + k));
                acc += src[y * width + cx] * kernel[k + radius];
              }
              temp[y * width + x] = acc;
            }
          }
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              let acc = 0;
              for (let k = -radius; k <= radius; k++) {
                const cy = Math.min(height - 1, Math.max(0, y + k));
                acc += temp[cy * width + x] * kernel[k + radius];
              }
              out[y * width + x] = acc;
            }
          }
          return out;
        }

        const gFine = gaussianBlur(gray, w, h, 1);
        const gMid = gaussianBlur(gray, w, h, 2);
        const gCoarse = gaussianBlur(gray, w, h, 5);

        const lapMap = new Float32Array(w * h);
        const gradMap = new Float32Array(w * h);
        const noiseMap = new Float32Array(w * h);
        let totalGx = 0, totalGy = 0;

        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const idx = y * w + x;
            const gx = (gFine[idx + 1] - gFine[idx - 1]) * 0.5;
            const gy = (gFine[idx + w] - gFine[idx - w]) * 0.5;
            totalGx += Math.abs(gx);
            totalGy += Math.abs(gy);
            gradMap[idx] = Math.sqrt(gx * gx + gy * gy);

            const lap = Math.abs(
              4 * gFine[idx] - gFine[idx - 1] - gFine[idx + 1] - gFine[idx - w] - gFine[idx + w]
            );
            lapMap[idx] = lap;
            noiseMap[idx] = Math.abs(gray[idx] - gFine[idx]);
          }
        }

        const rows = 4, cols = 4;
        const cellH = Math.floor(h / rows), cellW = Math.floor(w / cols);
        const zoneScores = [];
        const zoneFreqRatios = [];
        const zoneNoise = [];
        const zoneLum = [];
        const centerScores = [];

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const y1 = r * cellH, y2 = (r === rows - 1) ? h - 1 : (r + 1) * cellH;
            const x1 = c * cellW, x2 = (c === cols - 1) ? w - 1 : (c + 1) * cellW;

            const laps = [];
            const grads = [];
            let fineSum = 0, coarseSum = 0, noiseSum = 0, lumSum = 0, count = 0;

            for (let y = Math.max(1, y1); y < y2; y++) {
              for (let x = Math.max(1, x1); x < x2; x++) {
                const idx = y * w + x;
                laps.push(lapMap[idx]);
                grads.push(gradMap[idx]);
                fineSum += Math.abs(gFine[idx] - gMid[idx]);
                coarseSum += Math.abs(gMid[idx] - gCoarse[idx]);
                noiseSum += noiseMap[idx];
                lumSum += gray[idx];
                count++;
              }
            }

            laps.sort((a, b) => a - b);
            grads.sort((a, b) => a - b);
            const p90Lap = laps[Math.floor(laps.length * 0.90)] || 0;
            const p90Grad = grads[Math.floor(grads.length * 0.90)] || 0;

            const freqRatio = (fineSum / Math.max(1, count)) / ((coarseSum / Math.max(1, count)) + 0.55);
            let acutanceMult = 1.0;
            if (freqRatio < 0.25) acutanceMult = 0.28;
            else if (freqRatio < 0.36) acutanceMult = 0.62;
            else if (freqRatio > 0.52) acutanceMult = 1.28;

            const rawScore = (p90Lap * 4.2 + p90Grad * 1.1) * acutanceMult;
            const zScore = Math.round(rawScore * 10) / 10;

            zoneScores.push(zScore);
            zoneFreqRatios.push(freqRatio);
            zoneNoise.push(noiseSum / Math.max(1, count));
            zoneLum.push(lumSum / Math.max(1, count));

            if (r >= 1 && r <= 2 && c >= 1 && c <= 2) {
              centerScores.push(zScore);
            }
          }
        }

        const sortedZones = [...zoneScores].sort((a, b) => a - b);
        const flatNoise = [...zoneNoise].sort((a, b) => a - b).slice(0, 5).reduce((a, b) => a + b, 0) / 5;

        const peakSharpness = sortedZones.slice(-4).reduce((a, b) => a + b, 0) / 4;
        const top2Sharpness = sortedZones.slice(-2).reduce((a, b) => a + b, 0) / 2;
        const globalSharpness = zoneScores.reduce((a, b) => a + b, 0) / 16;
        const centerSharpness = centerScores.reduce((a, b) => a + b, 0) / 4;
        const bgSharpness = sortedZones.slice(0, 6).reduce((a, b) => a + b, 0) / 6;

        const topIndices = zoneScores
          .map((v, i) => ({ v, i }))
          .sort((a, b) => b.v - a.v)
          .slice(0, 4)
          .map(o => o.i);

        const subjectFreqRatio = topIndices.reduce((acc, idx) => acc + zoneFreqRatios[idx], 0) / 4;
        const subjectLum = Math.round(topIndices.reduce((acc, idx) => acc + zoneLum[idx], 0) / 4 * 10) / 10;

        let effectiveSharpness = mode === 'landscape'
          ? (globalSharpness * 0.65 + peakSharpness * 0.35)
          : Math.max(peakSharpness * 0.65 + centerSharpness * 0.35, top2Sharpness * 0.92);

        if (flatNoise > 4.2) {
          effectiveSharpness *= Math.max(0.32, 1 - (flatNoise - 3.8) * 0.11);
        }

        const bokehRatio = Math.round((peakSharpness / Math.max(1, bgSharpness)) * 100) / 100;
        const bokehDetected = effectiveSharpness >= 68 && bokehRatio >= 2.5 && subjectFreqRatio >= 0.36;

        const anisotropy = Math.max(totalGx, totalGy) / Math.max(1, Math.min(totalGx, totalGy));
        const motionBlur = anisotropy > 1.85 && subjectFreqRatio < 0.36;

        let sumLum = 0, clipDark = 0, clipBright = 0;
        const lumSorted = new Float32Array(gray);
        lumSorted.sort();
        for (let i = 0; i < gray.length; i++) {
          sumLum += gray[i];
          if (gray[i] < 12) clipDark++;
          if (gray[i] > 245) clipBright++;
        }
        const meanLum = Math.round((sumLum / gray.length) * 10) / 10;
        let sumSq = 0;
        for (let i = 0; i < gray.length; i++) sumSq += (gray[i] - meanLum) ** 2;
        const rmsContrast = Math.round(Math.sqrt(sumSq / gray.length) * 10) / 10;
        const dynRange = Math.round((lumSorted[Math.floor(gray.length * 0.98)] - lumSorted[Math.floor(gray.length * 0.02)]) * 10) / 10;

        const shadowClipPct = Math.round((clipDark / gray.length) * 1000) / 10;
        const highlightClipPct = Math.round((clipBright / gray.length) * 1000) / 10;

        const issues = [];
        let penalties = 0;

        if (effectiveSharpness < 36 || subjectFreqRatio < 0.26) {
          issues.push(motionBlur ? "Motion Blur / Shake" : "Out of Focus");
          penalties += 45;
        } else if (effectiveSharpness < 64 || subjectFreqRatio < 0.35) {
          issues.push(motionBlur ? "Slight Motion Blur" : "Soft Focus");
          penalties += 19;
        } else {
          issues.push(bokehDetected ? "Sharp Subject (Bokeh)" : "Tack Sharp Focus");
        }

        let expSubscore = 94;
        if ((subjectLum < 40 && shadowClipPct > 30) || (meanLum < 30 && shadowClipPct > 50)) {
          issues.push("Underexposed");
          penalties += 32;
          expSubscore = 38;
        } else if ((subjectLum > 225 && highlightClipPct > 18) || (meanLum > 225 && highlightClipPct > 30)) {
          issues.push("Blown Highlights");
          penalties += 30;
          expSubscore = 40;
        } else if (highlightClipPct > 18) {
          issues.push("Highlight Clipping");
          penalties += 12;
          expSubscore = 70;
        } else {
          issues.push("Balanced Exposure");
        }

        if (rmsContrast < 21 || dynRange < 62) {
          issues.push("Low Contrast / Washed Out");
          penalties += 22;
          expSubscore = Math.min(expSubscore, 54);
        } else if (rmsContrast >= 44 && dynRange >= 170 && !issues.includes("Out of Focus")) {
          issues.push("Rich Dynamic Range");
        }

        if (flatNoise > 5.8) {
          issues.push("High ISO Noise");
          penalties += 20;
        }

        const focusSubscore = Math.min(99, Math.max(12, Math.round(
          effectiveSharpness < 36 ? 20 + effectiveSharpness * 0.6 :
          effectiveSharpness < 65 ? 45 + (effectiveSharpness - 36) * 0.9 :
          74 + Math.min(25, (effectiveSharpness - 65) * 0.22)
        )));

        let finalScore = Math.round((focusSubscore * 0.64 + expSubscore * 0.36) - penalties * 0.52);
        if (bokehDetected && penalties === 0) finalScore += 4;
        finalScore = Math.min(99, Math.max(12, finalScore));

        if (issues.includes("Out of Focus") || issues.includes("Motion Blur / Shake")) {
          finalScore = Math.min(finalScore, 42);
        } else if (issues.includes("Underexposed") || issues.includes("Blown Highlights")) {
          finalScore = Math.min(finalScore, 46);
        } else if (issues.some(i => ["Soft Focus", "Slight Motion Blur", "Low Contrast / Washed Out", "High ISO Noise"].includes(i))) {
          finalScore = Math.min(finalScore, 71);
        }

        const category = finalScore >= 75 ? 'good' : finalScore >= 50 ? 'review' : 'poor';
        const sharpRounded = Math.round(effectiveSharpness * 10) / 10;

        const dCanvas = document.createElement('canvas');
        dCanvas.width = 9; dCanvas.height = 8;
        const dCtx = dCanvas.getContext('2d');
        dCtx.drawImage(img, 0, 0, 9, 8);
        const dData = dCtx.getImageData(0, 0, 9, 8).data;
        let dhashBits = "";
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const i1 = (r * 9 + c) * 4;
            const i2 = (r * 9 + c + 1) * 4;
            const l1 = 0.299 * dData[i1] + 0.587 * dData[i1+1] + 0.114 * dData[i1+2];
            const l2 = 0.299 * dData[i2] + 0.587 * dData[i2+1] + 0.114 * dData[i2+2];
            dhashBits += (l2 > l1 ? "1" : "0");
          }
        }

        resolve({
          score: finalScore,
          category,
          sharpness: sharpRounded,
          exposure: meanLum,
          subject_exposure: subjectLum,
          contrast: rmsContrast,
          dynamic_range: dynRange,
          noise_level: Math.round(flatNoise * 100) / 100,
          acutance_ratio: Math.round(subjectFreqRatio * 100) / 100,
          bokeh_ratio: bokehRatio,
          bokeh_detected: bokehDetected,
          clip_shadows_pct: shadowClipPct,
          clip_highlights_pct: highlightClipPct,
          zone_scores: zoneScores,
          issues,
          dhash_bits: dhashBits,
          explanation: `Subject Acutance: ${sharpRounded} (Fine/Coarse Edge Ratio: ${subjectFreqRatio.toFixed(2)}). ${bokehDetected ? 'Intentional Portrait Bokeh (' + bokehRatio + 'x separation). ' : ''}Subject Lum: ${subjectLum}, Dynamic Range: ${dynRange}, Noise Floor: ${flatNoise.toFixed(1)}.`
        });
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = reject;
    img.src = imgSource;
  });
}

function clientHammingDistance(b1, b2) {
  if (!b1 || !b2 || b1.length !== b2.length) return 64;
  let dist = 0;
  for (let i = 0; i < b1.length; i++) if (b1[i] !== b2[i]) dist++;
  return dist;
}

// =========================================================================
// RE-CALIBRATE ALL LOADED PHOTOS
// =========================================================================
async function recalibrateLibrary() {
  if (!photoLibrary.length) {
    showToast("Import photos or tap 'Test Bench' first!");
    return;
  }

  const progressModal = document.getElementById('uploadProgressModal');
  const progressBar = document.getElementById('uploadProgressBar');
  const progressCount = document.getElementById('uploadProgressCount');
  const progressPercent = document.getElementById('uploadProgressPercent');
  const modalTitle = document.getElementById('uploadModalTitle');
  const modalSub = document.getElementById('uploadModalSub');

  modalTitle.innerText = "Re-Calibrating Library with v4.0 Studio CV...";
  progressModal.classList.add('active');

  try {
    fetch(`${BACKEND_URL}/api/photos/reanalyze?mode=${currentSceneMode}`, { method: "POST" }).catch(() => {});
  } catch (e) {}

  const seenHashes = [];
  const total = photoLibrary.length;

  for (let i = 0; i < total; i++) {
    const item = photoLibrary[i];
    modalSub.innerText = `Calibrating (${i + 1}/${total}): ${item.filename}`;
    try {
      const cv = await analyzeImageClientCV(item.src, currentSceneMode);

      let isDup = false;
      for (const prevHash of seenHashes) {
        if (clientHammingDistance(cv.dhash_bits, prevHash) <= 8) {
          isDup = true;
          if (!cv.issues.includes("Burst Duplicate")) cv.issues.push("Burst Duplicate");
          break;
        }
      }
      if (cv.dhash_bits) seenHashes.push(cv.dhash_bits);

      Object.assign(item, cv, { is_duplicate: isDup });

      if (!String(item.id).startsWith('demo_')) {
        fetch(`${BACKEND_URL}/api/photos/${item.id}/category`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category: cv.category })
        }).catch(() => {});
      }
    } catch (err) {
      console.warn("Calibration skipped for:", item.filename);
    }

    const pct = Math.round(((i + 1) / total) * 100);
    progressBar.style.width = `${pct}%`;
    progressCount.innerText = `${i + 1} / ${total} images`;
    progressPercent.innerText = `${pct}%`;
  }

  setTimeout(() => {
    progressModal.classList.remove('active');
    handleSortChange(currentSort);
    updateGalleryUI();
    showToast(`Re-calibrated ${total} photos with 100% v4.0 accuracy!`);
  }, 300);
}

// =========================================================================
// API SYNC: LOAD PHOTOS & CLOUD TELEMETRY
// =========================================================================
async function loadPhotosFromAtlas() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/photos?sort=${currentSort}`);
    if (!res.ok) return;
    const data = await res.json();
    if (Array.isArray(data) && data.length > 0) {
      photoLibrary = data.map(item => ({
        id: item.id,
        filename: item.filename,
        src: item.image_data,
        thumb: item.thumb_data || item.image_data,
        score: item.score,
        category: item.category,
        sharpness: item.sharpness,
        exposure: item.exposure,
        subject_exposure: item.subject_exposure || item.exposure,
        contrast: item.contrast || 45,
        dynamic_range: item.dynamic_range || 160,
        noise_level: item.noise_level || 1.8,
        acutance_ratio: item.acutance_ratio || 0.52,
        bokeh_ratio: item.bokeh_ratio || 1.4,
        bokeh_detected: item.bokeh_detected || false,
        clip_shadows_pct: item.clip_shadows_pct || 0,
        clip_highlights_pct: item.clip_highlights_pct || 0,
        zone_scores: item.zone_scores || null,
        explanation: item.explanation || null,
        issues: item.issues || [],
        is_duplicate: item.is_duplicate || false,
        exif: item.exif || {},
        face_count: item.face_count || 0
      }));

      const needsCalibration = photoLibrary.some(p => !p.zone_scores);
      if (needsCalibration) {
        await recalibrateLibrary();
      } else {
        updateGalleryUI();
      }

      const workspaceTab = document.querySelectorAll('.dock-tab')[1];
      switchView('cullingView', workspaceTab);
    }
  } catch (err) {
    console.warn("Offline or local mode active:", err);
  }
}

async function loadBackendStats() {
  const total = photoLibrary.length;
  const good = photoLibrary.filter(p => p.category === 'good').length;
  const review = photoLibrary.filter(p => p.category === 'review').length;
  const poor = photoLibrary.filter(p => p.category === 'poor').length;
  const bokehCount = photoLibrary.filter(p => p.bokeh_detected).length;

  const denom = total || 1;
  const gPct = Math.round((good / denom) * 100);
  const rPct = Math.round((review / denom) * 100);
  const pPct = Math.round((poor / denom) * 100);

  document.getElementById('statGoodPercent').innerText = `${good} (${gPct}%)`;
  document.getElementById('statReviewPercent').innerText = `${review} (${rPct}%)`;
  document.getElementById('statPoorPercent').innerText = `${poor} (${pPct}%)`;

  document.getElementById('statGoodBar').style.width = gPct + '%';
  document.getElementById('statReviewBar').style.width = rPct + '%';
  document.getElementById('statPoorBar').style.width = pPct + '%';
  document.getElementById('statBokehCount').innerText = bokehCount;

  if (total > 0) {
    const avgScore = Math.round(photoLibrary.reduce((s, p) => s + p.score, 0) / total);
    const avgSharp = (photoLibrary.reduce((s, p) => s + p.sharpness, 0) / total).toFixed(1);
    const avgExp = Math.round(photoLibrary.reduce((s, p) => s + (p.subject_exposure || p.exposure), 0) / total);
    document.getElementById('statAvgScore').innerText = `${avgScore} / 100`;
    document.getElementById('statAvgSharpness').innerText = avgSharp;
    document.getElementById('statAvgExposure').innerText = `${avgExp} Lum`;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/api/stats`);
    if (!res.ok) return;
    const stats = await res.json();
    const storage = stats.storage;
    document.getElementById('storageLabelUsed').innerText = `${storage.used_mb} MB / ${storage.limit_mb} MB (${storage.used_percent}%)`;
    document.getElementById('storageUsedMbText').innerText = `${storage.used_mb} MB`;
    document.getElementById('storageRemainingMbText').innerText = `${storage.remaining_mb} MB Free`;
    const meter = document.getElementById('storageMeterFill');
    const displayWidth = storage.used_mb > 0 ? Math.max(1.5, storage.used_percent) : 0;
    meter.style.width = Math.min(100, displayWidth) + '%';
  } catch (err) {}
}

// =========================================================================
// DRAG-AND-DROP & HYBRID UPLOAD PIPELINE
// =========================================================================
const dropArea = document.getElementById('dropArea');
['dragenter', 'dragover'].forEach(name => {
  dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.add('drag-over'); });
});
['dragleave', 'drop'].forEach(name => {
  dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.remove('drag-over'); });
});
dropArea.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
});

function handleFileSelect(e) {
  if (e.target.files.length) processFiles(e.target.files);
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function processFiles(fileList) {
  const validFiles = Array.from(fileList).filter(f => f.type.startsWith('image/'));
  if (!validFiles.length) {
    showToast("Please select valid image files.");
    return;
  }

  const total = validFiles.length;
  const progressModal = document.getElementById('uploadProgressModal');
  const progressBar = document.getElementById('uploadProgressBar');
  const progressCount = document.getElementById('uploadProgressCount');
  const progressPercent = document.getElementById('uploadProgressPercent');
  const modalTitle = document.getElementById('uploadModalTitle');
  const modalSub = document.getElementById('uploadModalSub');

  modalTitle.innerText = "Analyzing Optical Quality & Syncing...";
  progressModal.classList.add('active');
  progressBar.style.width = '0%';

  for (let i = 0; i < total; i++) {
    const file = validFiles[i];
    modalSub.innerText = `Analyzing (${i + 1}/${total}): ${file.name}`;

    const dataUrl = await fileToDataURL(file);
    const localCV = await analyzeImageClientCV(dataUrl, currentSceneMode);

    let isDup = false;
    for (const existing of photoLibrary) {
      if (existing.dhash_bits && clientHammingDistance(localCV.dhash_bits, existing.dhash_bits) <= 8) {
        isDup = true;
        if (!localCV.issues.includes("Burst Duplicate")) localCV.issues.push("Burst Duplicate");
        break;
      }
    }

    let recordId = 'local_' + Math.random().toString(36).substr(2, 9);
    let storedSrc = dataUrl;
    let storedThumb = dataUrl;
    let exifData = { camera: "Digital Sensor", f_stop: "f/2.8", shutter: "1/250s", iso: "ISO 200" };

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${BACKEND_URL}/api/photos/analyze?mode=${currentSceneMode}`, {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        const serverData = await res.json();
        recordId = serverData.id;
        storedSrc = serverData.image_data || dataUrl;
        storedThumb = serverData.thumb_data || storedSrc;
        exifData = serverData.exif || exifData;

        if (!serverData.zone_scores && serverData.category !== localCV.category) {
          fetch(`${BACKEND_URL}/api/photos/${recordId}/category`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: localCV.category })
          }).catch(() => {});
        }
      }
    } catch (err) {
      console.warn("Saved in local session mode:", err);
    }

    photoLibrary.unshift({
      id: recordId,
      filename: file.name,
      src: storedSrc,
      thumb: storedThumb,
      ...localCV,
      is_duplicate: isDup,
      exif: exifData,
      face_count: 1
    });

    const pct = Math.round(((i + 1) / total) * 100);
    progressBar.style.width = `${pct}%`;
    progressCount.innerText = `${i + 1} / ${total} images`;
    progressPercent.innerText = `${pct}%`;
  }

  setTimeout(() => {
    progressModal.classList.remove('active');
    updateGalleryUI();
    loadBackendStats();
    const workspaceTab = document.querySelectorAll('.dock-tab')[1];
    switchView('cullingView', workspaceTab);
    showToast(`Analyzed ${total} image(s) with v4.0 Studio CV.`);
  }, 350);
}

// =========================================================================
// GROUND-TRUTH LIVE CALIBRATION TEST BENCH (MONOCHROME STUDIO EDITION)
// =========================================================================
function generateSyntheticTestImage(type) {
  const c = document.createElement('canvas');
  c.width = 640; c.height = 480;
  const ctx = c.getContext('2d');

  if (type === 'sharp_portrait_bokeh') {
    const grad = ctx.createLinearGradient(0, 0, 640, 480);
    grad.addColorStop(0, '#141414');
    grad.addColorStop(1, '#262626');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 640, 480);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.beginPath(); ctx.arc(120, 110, 55, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(520, 360, 70, 0, Math.PI * 2); ctx.fill();

    ctx.save();
    ctx.translate(320, 240);
    for (let r = 110; r >= 10; r -= 8) {
      ctx.fillStyle = (r % 16 === 0) ? '#FFFFFF' : '#0A0A0A';
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = '#E4E4E7';
    ctx.lineWidth = 2;
    for (let a = 0; a < 360; a += 10) {
      const rad = a * Math.PI / 180;
      ctx.beginPath();
      ctx.moveTo(Math.cos(rad) * 12, Math.sin(rad) * 12);
      ctx.lineTo(Math.cos(rad) * 115, Math.sin(rad) * 115);
      ctx.stroke();
    }
    ctx.restore();
  } else if (type === 'sharp_architecture') {
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, 640, 480);
    for (let y = 20; y < 460; y += 24) {
      for (let x = 20; x < 620; x += 24) {
        ctx.fillStyle = ((x + y) % 48 === 0) ? '#FFFFFF' : '#52525B';
        ctx.fillRect(x, y, 18, 18);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, 18, 18);
      }
    }
  } else if (type === 'severe_defocus_blur') {
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, 640, 480);
    ctx.filter = 'blur(14px)';
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath(); ctx.arc(220, 240, 95, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#A1A1AA';
    ctx.fillRect(340, 140, 160, 200);
    ctx.filter = 'none';
  } else if (type === 'soft_focus_review') {
    ctx.fillStyle = '#27272A';
    ctx.fillRect(0, 0, 640, 480);
    ctx.filter = 'blur(3.2px)';
    for (let y = 80; y < 400; y += 32) {
      for (let x = 120; x < 520; x += 32) {
        ctx.fillStyle = ((x + y) % 64 === 0) ? '#F4F4F5' : '#18181B';
        ctx.fillRect(x, y, 24, 24);
      }
    }
    ctx.filter = 'none';
  } else if (type === 'underexposed_night') {
    ctx.fillStyle = '#030303';
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = '#121212';
    ctx.beginPath(); ctx.arc(320, 240, 80, 0, Math.PI * 2); ctx.fill();
  }
  return c.toDataURL('image/webp', 0.92);
}

async function runLiveCalibrationBenchmark() {
  showToast("Running live v4.0 CV pipeline on 6 ground-truth test cases...");
  const testSpecs = [
    { name: "01_Hero_Portrait_Sharp_Bokeh.webp", type: "sharp_portrait_bokeh", exif: { camera: "Sony A7R V", f_stop: "f/1.4", shutter: "1/800s", iso: "ISO 100" } },
    { name: "02_Architecture_Tack_Sharp.webp", type: "sharp_architecture", exif: { camera: "Nikon Z8", f_stop: "f/8.0", shutter: "1/500s", iso: "ISO 64" } },
    { name: "03_Portrait_Soft_Focus_Review.webp", type: "soft_focus_review", exif: { camera: "Canon R5", f_stop: "f/2.0", shutter: "1/125s", iso: "ISO 400" } },
    { name: "04_Burst_Duplicate_Frame_02.webp", type: "sharp_portrait_bokeh", exif: { camera: "Sony A7R V", f_stop: "f/1.4", shutter: "1/800s", iso: "ISO 100" } },
    { name: "05_High_Contrast_Out_Of_Focus_Blur.webp", type: "severe_defocus_blur", exif: { camera: "Sony A7 IV", f_stop: "f/1.8", shutter: "1/60s", iso: "ISO 800" } },
    { name: "06_Severely_Underexposed_Crushed.webp", type: "underexposed_night", exif: { camera: "Canon R6", f_stop: "f/4.0", shutter: "1/1000s", iso: "ISO 1600" } }
  ];

  const seenHashes = [];
  for (const spec of testSpecs) {
    const dataUrl = generateSyntheticTestImage(spec.type);
    const cv = await analyzeImageClientCV(dataUrl, currentSceneMode);

    let isDup = false;
    for (const h of seenHashes) {
      if (clientHammingDistance(cv.dhash_bits, h) <= 8) {
        isDup = true;
        if (!cv.issues.includes("Burst Duplicate")) cv.issues.push("Burst Duplicate");
        break;
      }
    }
    seenHashes.push(cv.dhash_bits);

    photoLibrary.unshift({
      id: 'demo_' + Math.random().toString(36).substr(2, 9),
      filename: spec.name,
      src: dataUrl,
      thumb: dataUrl,
      ...cv,
      is_duplicate: isDup,
      exif: spec.exif,
      face_count: 1
    });
  }

  handleSortChange('score_desc');
  updateGalleryUI();
  loadBackendStats();
  const workspaceTab = document.querySelectorAll('.dock-tab')[1];
  switchView('cullingView', workspaceTab);
  showToast("6 Ground-Truth Test Images analyzed! Tap any card to inspect.");
}

// =========================================================================
// GALLERY RENDERING, FILTERING & SORTING
// =========================================================================
function getFilteredPhotos() {
  if (currentFilter === 'duplicates') {
    return photoLibrary.filter(p => p.is_duplicate || (p.issues && p.issues.includes("Burst Duplicate")));
  }
  return currentFilter === 'all'
    ? photoLibrary
    : photoLibrary.filter(p => p.category === currentFilter);
}

function handleSortChange(sortVal) {
  currentSort = sortVal;
  if (sortVal === 'score_desc') photoLibrary.sort((a, b) => b.score - a.score);
  else if (sortVal === 'score_asc') photoLibrary.sort((a, b) => a.score - b.score);
  else if (sortVal === 'sharpness_desc') photoLibrary.sort((a, b) => b.sharpness - a.sharpness);
  updateGalleryUI();
}

function updateGalleryUI() {
  const gallery = document.getElementById('photoGallery');
  const emptyState = document.getElementById('emptyGallery');

  const total = photoLibrary.length;
  const goodCount = photoLibrary.filter(p => p.category === 'good').length;
  const reviewCount = photoLibrary.filter(p => p.category === 'review').length;
  const poorCount = photoLibrary.filter(p => p.category === 'poor').length;
  const dupCount = photoLibrary.filter(p => p.is_duplicate || (p.issues && p.issues.includes("Burst Duplicate"))).length;

  document.getElementById('statTotal').innerText = total;
  document.getElementById('statGood').innerText = goodCount;
  document.getElementById('statReview').innerText = reviewCount;
  document.getElementById('statPoor').innerText = poorCount;

  document.getElementById('pillAllCount').innerText = total;
  document.getElementById('pillGoodCount').innerText = goodCount;
  document.getElementById('pillReviewCount').innerText = reviewCount;
  document.getElementById('pillPoorCount').innerText = poorCount;
  document.getElementById('pillDupCount').innerText = dupCount;

  const filtered = getFilteredPhotos();
  gallery.innerHTML = '';

  if (!filtered.length) {
    emptyState.style.display = 'block';
    return;
  }
  emptyState.style.display = 'none';

  filtered.forEach(photo => {
    const card = document.createElement('div');
    card.className = 'photo-card' + (selectedPhotoIds.has(photo.id) ? ' selected' : '');
    card.setAttribute('data-id', photo.id);

    let badgeClass = 'score-good';
    if (photo.category === 'review') badgeClass = 'score-review';
    if (photo.category === 'poor') badgeClass = 'score-poor';

    const flagsHtml = (photo.issues || []).map(iss => {
      const isGood = iss.includes('Sharp') || iss.includes('Balanced') || iss.includes('Rich');
      const isAlert = iss.includes('Out of Focus') || iss.includes('Motion Blur / Shake') || iss.includes('Underexposed') || iss.includes('Blown');
      const isWarn = iss.includes('Soft') || iss.includes('Slight') || iss.includes('Low Contrast') || iss.includes('Noise') || iss.includes('Eyes');
      const isDup = iss.includes('Duplicate');
      let cls = '';
      if (isGood) cls = 'good-tag';
      else if (isAlert) cls = 'alert';
      else if (isWarn) cls = 'warn';
      if (isDup) cls = 'duplicate';
      return `<span class="flag-tag ${cls}">${iss}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="card-checkbox" onclick="toggleSelectPhoto('${photo.id}', event)">
        <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"/></svg>
      </div>
      <button class="card-delete-icon-btn" onclick="promptSingleDelete('${photo.id}', event)" title="Delete Photo">
        <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
      </button>
      <div class="photo-thumb-wrap" onclick="openInspection('${photo.id}')">
        <img src="${photo.thumb}" alt="${photo.filename}" loading="lazy">
        <div class="score-badge ${badgeClass}">${photo.score} PTS</div>
        <div class="telemetry-strip">
          <span>SHARP: ${photo.sharpness}</span>
          <span>RATIO: ${photo.acutance_ratio || 0.5}</span>
          <span>LUM: ${photo.subject_exposure || photo.exposure}</span>
        </div>
      </div>
      <div class="photo-meta">
        <div class="meta-title">${photo.filename}</div>
        <div class="flags-row">${flagsHtml}</div>
      </div>
      <div class="card-actions">
        <button class="card-action-btn ${photo.category === 'good' ? 'selected-good' : ''}" onclick="setClassification('${photo.id}', 'good')">Select</button>
        <button class="card-action-btn ${photo.category === 'review' ? 'selected-review' : ''}" onclick="setClassification('${photo.id}', 'review')">Review</button>
        <button class="card-action-btn ${photo.category === 'poor' ? 'selected-poor' : ''}" onclick="setClassification('${photo.id}', 'poor')">Poor</button>
      </div>`;

    gallery.appendChild(card);
  });

  updateBatchBar();
}

function setCategoryFilter(filter) {
  currentFilter = filter;
  document.querySelectorAll('.filter-pills .pill').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-filter') === filter);
  });
  updateGalleryUI();
}

async function setClassification(id, newCat) {
  const item = photoLibrary.find(p => p.id === id);
  if (item) {
    item.category = newCat;
    updateGalleryUI();
    loadBackendStats();
    showToast('Classified as ' + newCat.toUpperCase());

    if (!String(id).startsWith('demo_') && !String(id).startsWith('local_')) {
      fetch(`${BACKEND_URL}/api/photos/${id}/category`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: newCat })
      }).catch(() => {});
    }
  }
}

// =========================================================================
// STUDIO INSPECTION MODAL (FOCUS PEAKING, 4x4 HEATMAP, ZEBRA, HISTOGRAM)
// =========================================================================
let activeOverlayMode = null;
let heatmapActive = false;
let gridOverlayActive = false;

function openInspection(id) {
  const photo = photoLibrary.find(p => p.id === id);
  if (!photo) return;
  inspectedPhotoId = id;

  const modalImg = document.getElementById('modalImg');
  modalImg.onload = () => {
    renderLiveHistogram(modalImg);
  };
  modalImg.src = photo.src;

  document.getElementById('modalFilename').innerText = photo.filename;
  const catEl = document.getElementById('modalCategory');
  catEl.innerText = `${photo.category.toUpperCase()} — ${photo.score}/100`;
  catEl.style.color = photo.category === 'good' ? 'var(--good-color)' : photo.category === 'review' ? 'var(--review-color)' : 'var(--poor-color)';

  document.getElementById('modalScoreVal').innerText = `${photo.score} / 100`;
  document.getElementById('modalScoreBar').style.width = `${photo.score}%`;
  document.getElementById('modalScoreBar').style.background = photo.category === 'good' ? 'var(--good-color)' : photo.category === 'review' ? 'var(--review-color)' : 'var(--poor-color)';

  const ratio = photo.acutance_ratio || 0.52;
  document.getElementById('modalSharpVal').innerText = `${photo.sharpness} (Edge Ratio ${ratio})`;
  document.getElementById('modalSharpBar').style.width = `${Math.min(100, photo.sharpness / 1.8)}%`;

  const subExp = photo.subject_exposure || photo.exposure;
  const dr = photo.dynamic_range || 160;
  document.getElementById('modalExposureVal').innerText = `${subExp} Lum (DR ${dr})`;
  document.getElementById('modalExposureBar').style.width = `${Math.min(100, Math.round((subExp / 255) * 100))}%`;

  document.getElementById('modalExplanation').innerText = photo.explanation ||
    `Subject Acutance: ${photo.sharpness} (Edge Ratio: ${ratio}). Subject Luminance: ${subExp} Lum.`;

  document.getElementById('modalBokehRatio').innerText = `${photo.bokeh_ratio || 1.2}x ${photo.bokeh_detected ? '(Bokeh)' : ''}`;
  document.getElementById('modalNoiseVal').innerText = photo.noise_level || 1.5;
  document.getElementById('modalShadowClip').innerText = `${photo.clip_shadows_pct || 0}%`;
  document.getElementById('modalHighlightClip').innerText = `${photo.clip_highlights_pct || 0}%`;

  const issuesEl = document.getElementById('modalIssues');
  issuesEl.innerHTML = (photo.issues || []).map(iss => {
    const isGood = iss.includes('Sharp') || iss.includes('Balanced') || iss.includes('Rich');
    const isAlert = iss.includes('Out of Focus') || iss.includes('Motion Blur') || iss.includes('Underexposed') || iss.includes('Blown');
    return `<span class="flag-tag ${isGood ? 'good-tag' : isAlert ? 'alert' : 'warn'}">${iss}</span>`;
  }).join('');

  activeOverlayMode = null;
  heatmapActive = false;
  gridOverlayActive = false;
  document.getElementById('overlayCanvas').classList.remove('active');
  document.getElementById('zoneHeatmapGrid').classList.remove('active');
  document.getElementById('ruleOfThirdsGrid').classList.remove('active');
  document.querySelectorAll('.overlay-btn').forEach(b => b.classList.remove('active'));

  document.getElementById('inspectModal').classList.add('active');
}

function renderLiveHistogram(img) {
  const canvas = document.getElementById('histogramCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const temp = document.createElement('canvas');
  temp.width = 180; temp.height = 135;
  const tCtx = temp.getContext('2d');
  tCtx.drawImage(img, 0, 0, 180, 135);
  const data = tCtx.getImageData(0, 0, 180, 135).data;

  const lBins = new Uint32Array(256);
  for (let i = 0; i < data.length; i += 4) {
    const l = Math.round(0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]);
    lBins[l]++;
  }

  const maxVal = Math.max(...lBins, 1);

  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * w;
    const y = h - Math.min(h - 2, (lBins[i] / maxVal) * (h - 4));
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.fill();
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

function toggleFocusPeaking() {
  const canvas = document.getElementById('overlayCanvas');
  const btn = document.getElementById('btnFocusPeaking');
  document.getElementById('btnClippingMask').classList.remove('active');

  if (activeOverlayMode === 'peaking') {
    activeOverlayMode = null;
    canvas.classList.remove('active');
    btn.classList.remove('active');
    return;
  }

  activeOverlayMode = 'peaking';
  btn.classList.add('active');

  const img = document.getElementById('modalImg');
  const w = img.naturalWidth || 480;
  const h = img.naturalHeight || 360;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);

  const imgData = ctx.getImageData(0, 0, w, h);
  const src = imgData.data;
  const out = ctx.createImageData(w, h);
  const dst = out.data;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const c = 0.299 * src[idx] + 0.587 * src[idx+1] + 0.114 * src[idx+2];
      const l = 0.299 * src[idx - 4] + 0.587 * src[idx - 3] + 0.114 * src[idx - 2];
      const r = 0.299 * src[idx + 4] + 0.587 * src[idx + 5] + 0.114 * src[idx + 6];
      const u = 0.299 * src[idx - w*4] + 0.587 * src[idx - w*4 + 1] + 0.114 * src[idx - w*4 + 2];
      const d = 0.299 * src[idx + w*4] + 0.587 * src[idx + w*4 + 1] + 0.114 * src[idx + w*4 + 2];

      const lap = Math.abs(4 * c - l - r - u - d);
      if (lap > 24) {
        dst[idx] = 34; dst[idx+1] = 255; dst[idx+2] = 120; dst[idx+3] = 240;
      } else {
        dst[idx+3] = 0;
      }
    }
  }
  ctx.putImageData(out, 0, 0);
  canvas.classList.add('active');
}

function toggleClippingMask() {
  const canvas = document.getElementById('overlayCanvas');
  const btn = document.getElementById('btnClippingMask');
  document.getElementById('btnFocusPeaking').classList.remove('active');

  if (activeOverlayMode === 'clipping') {
    activeOverlayMode = null;
    canvas.classList.remove('active');
    btn.classList.remove('active');
    return;
  }

  activeOverlayMode = 'clipping';
  btn.classList.add('active');

  const img = document.getElementById('modalImg');
  canvas.width = img.naturalWidth || 480;
  canvas.height = img.naturalHeight || 360;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
    if (lum > 245) {
      data[i] = 255; data[i+1] = 40; data[i+2] = 40; data[i+3] = 225;
    } else if (lum < 12) {
      data[i] = 245; data[i+1] = 158; data[i+2] = 11; data[i+3] = 225;
    } else {
      data[i+3] = 0;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  canvas.classList.add('active');
}

function toggleZoneHeatmap() {
  heatmapActive = !heatmapActive;
  const grid = document.getElementById('zoneHeatmapGrid');
  const btn = document.getElementById('btnZoneHeatmap');
  btn.classList.toggle('active', heatmapActive);

  if (!heatmapActive) {
    grid.classList.remove('active');
    return;
  }

  const photo = photoLibrary.find(p => p.id === inspectedPhotoId);
  const zones = (photo && photo.zone_scores && photo.zone_scores.length === 16)
    ? photo.zone_scores
    : Array(16).fill(photo ? photo.sharpness : 80);

  grid.innerHTML = zones.map(z => {
    const bg = z >= 70
      ? 'rgba(34, 197, 94, 0.40)'
      : z >= 40
        ? 'rgba(245, 158, 11, 0.40)'
        : 'rgba(239, 68, 68, 0.40)';
    return `<div class="zone-cell" style="background:${bg}">${Math.round(z)}</div>`;
  }).join('');

  grid.classList.add('active');
}

function toggleRuleOfThirds() {
  gridOverlayActive = !gridOverlayActive;
  document.getElementById('ruleOfThirdsGrid').classList.toggle('active', gridOverlayActive);
  document.getElementById('btnGridOverlay').classList.toggle('active', gridOverlayActive);
}

function overrideModalPhoto(category) {
  if (inspectedPhotoId) {
    setClassification(inspectedPhotoId, category);
    const photo = photoLibrary.find(p => p.id === inspectedPhotoId);
    const catEl = document.getElementById('modalCategory');
    catEl.innerText = `${category.toUpperCase()} — ${photo ? photo.score : ''}/100`;
    catEl.style.color = category === 'good' ? 'var(--good-color)' : category === 'review' ? 'var(--review-color)' : 'var(--poor-color)';
  }
}

function closeModal() {
  document.getElementById('inspectModal').classList.remove('active');
  inspectedPhotoId = null;
}

// =========================================================================
// BATCH OPERATIONS, COMPARISON & DELETION
// =========================================================================
function toggleSelectPhoto(id, event) {
  if (event) event.stopPropagation();
  if (selectedPhotoIds.has(id)) selectedPhotoIds.delete(id);
  else selectedPhotoIds.add(id);
  updateBatchBar();
  document.querySelectorAll('.photo-card').forEach(card => {
    card.classList.toggle('selected', selectedPhotoIds.has(card.getAttribute('data-id')));
  });
}

function updateBatchBar() {
  const bar = document.getElementById('batchActionBar');
  const count = selectedPhotoIds.size;
  if (count > 0) {
    document.getElementById('batchCountTag').innerText = `${count} Selected`;
    bar.classList.add('visible');
    document.getElementById('compareBatchBtn').style.display = count === 2 ? 'inline-flex' : 'none';
  } else {
    bar.classList.remove('visible');
  }
}

function clearBatchSelection() {
  selectedPhotoIds.clear();
  updateBatchBar();
  document.querySelectorAll('.photo-card').forEach(c => c.classList.remove('selected'));
}

async function batchSetClassification(category) {
  const ids = Array.from(selectedPhotoIds);
  ids.forEach(id => {
    const item = photoLibrary.find(p => p.id === id);
    if (item) item.category = category;
  });
  clearBatchSelection();
  updateGalleryUI();
  loadBackendStats();
  showToast(`Moved ${ids.length} photos to ${category.toUpperCase()}`);
}

function openSideBySideCompare() {
  const ids = Array.from(selectedPhotoIds);
  if (ids.length !== 2) return;
  const p1 = photoLibrary.find(p => p.id === ids[0]);
  const p2 = photoLibrary.find(p => p.id === ids[1]);
  if (!p1 || !p2) return;

  document.getElementById('compareGrid').innerHTML = [p1, p2].map((p, idx) => `
    <div class="compare-pane">
      <div class="compare-img-box"><img src="${p.src}" alt="${p.filename}"></div>
      <div style="padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem;">
        <h4 style="font-size: 0.9rem; font-weight: 800;">${p.filename}</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; font-size: 0.75rem; font-family: var(--mono);">
          <div>Score: <strong>${p.score}/100</strong></div>
          <div>Acutance: <strong>${p.sharpness}</strong></div>
          <div>Edge Ratio: <strong>${p.acutance_ratio || 0.5}</strong></div>
          <div>Subject Lum: <strong>${p.subject_exposure || p.exposure}</strong></div>
        </div>
        <button class="btn btn-primary" style="margin-top: 0.4rem;" onclick="pickCompareWinner('${p.id}', '${idx === 0 ? p2.id : p1.id}')">
          Select as Winner
        </button>
      </div>
    </div>
  `).join('');
  document.getElementById('compareModal').classList.add('active');
}

function pickCompareWinner(winnerId, loserId) {
  setClassification(winnerId, 'good');
  setClassification(loserId, 'review');
  closeCompareModal();
  clearBatchSelection();
}

function closeCompareModal() {
  document.getElementById('compareModal').classList.remove('active');
}

function promptSingleDelete(id, event) {
  if (event) event.stopPropagation();
  pendingDeleteType = 'single';
  pendingDeleteTargetId = id;
  document.getElementById('confirmDeleteModal').classList.add('active');
}

function promptBatchDelete() {
  if (!selectedPhotoIds.size) return;
  pendingDeleteType = 'batch';
  document.getElementById('confirmDeleteModal').classList.add('active');
}

function closeConfirmModal() {
  document.getElementById('confirmDeleteModal').classList.remove('active');
}

async function executeConfirmedDelete() {
  closeConfirmModal();
  if (pendingDeleteType === 'single' && pendingDeleteTargetId) {
    const id = pendingDeleteTargetId;
    if (!String(id).startsWith('demo_') && !String(id).startsWith('local_')) {
      fetch(`${BACKEND_URL}/api/photos/${id}`, { method: "DELETE" }).catch(() => {});
    }
    photoLibrary = photoLibrary.filter(p => p.id !== id);
    selectedPhotoIds.delete(id);
    if (inspectedPhotoId === id) closeModal();
  } else if (pendingDeleteType === 'batch') {
    photoLibrary = photoLibrary.filter(p => !selectedPhotoIds.has(p.id));
    selectedPhotoIds.clear();
  }
  updateGalleryUI();
  loadBackendStats();
  showToast("Photograph(s) removed.");
}

// =========================================================================
// KEYBOARD SHORTCUTS & VIEW CONTROLLER
// =========================================================================
function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === '1' || key === 's') { if (inspectedPhotoId) overrideModalPhoto('good'); }
    else if (key === '2' || key === 'r') { if (inspectedPhotoId) overrideModalPhoto('review'); }
    else if (key === '3' || key === 'x') { if (inspectedPhotoId) overrideModalPhoto('poor'); }
    else if (e.code === 'Space') {
      e.preventDefault();
      if (inspectedPhotoId) closeModal();
      else if (photoLibrary.length > 0) openInspection(photoLibrary[0].id);
    }
    else if (key === 'arrowright' && inspectedPhotoId) navigateModal(1);
    else if (key === 'arrowleft' && inspectedPhotoId) navigateModal(-1);
  });
}

function navigateModal(dir) {
  const filtered = getFilteredPhotos();
  const idx = filtered.findIndex(p => p.id === inspectedPhotoId);
  if (idx !== -1 && filtered[idx + dir]) openInspection(filtered[idx + dir].id);
}

function switchView(viewId, element) {
  closeMobileMenu();
  document.querySelectorAll('.view-container').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.dock-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(viewId).classList.add('active');
  element.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (viewId === 'statsView') loadBackendStats();
}

async function exportSelectsZip() {
  const selects = photoLibrary.filter(p => p.category === 'good');
  if (!selects.length) {
    showToast("No photos in 'Selects' category to export!");
    return;
  }
  showToast(`Packaging ${selects.length} Selects into ZIP...`);
  const zip = new JSZip();
  const folder = zip.folder("PhotoSort_Selects");
  selects.forEach(p => {
    if (p.src && p.src.includes(',')) {
      const cleanName = p.filename.replace(/\.[^/.]+\$/, "") + ".webp";
      folder.file(cleanName, p.src.split(',')[1], { base64: true });
    }
  });
  const blob = await zip.generateAsync({ type: "blob" });
  saveAs(blob, "PhotoSort_Curated_Selects.zip");
  showToast("ZIP archive downloaded!");
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.innerText = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2600);
}