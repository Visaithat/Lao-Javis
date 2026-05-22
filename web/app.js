// Javis visualizer: 5 dark pill-shaped bars that animate per state.
// Python pushes state and mic amplitude via window.setState() / window.feedAmplitude()
// invoked through pywebview's evaluate_js bridge.

(() => {
  const canvas = document.getElementById('viz');
  const ctx = canvas.getContext('2d', { alpha: true });

  // Crisp on HiDPI: scale backing store by devicePixelRatio
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- visual config ---
  const BAR_COUNT = 5;
  const BAR_WIDTH = 26;
  const BAR_SPACING = 16;
  const BAR_COLOR = '#3a3a3a';
  const MIN_H = 28;
  const MAX_H = 180;
  const SMOOTH = 0.28;
  const AMP_DECAY = 0.85;

  // --- state ---
  let state = 'idle';
  let amp = 0;
  const cur = new Array(BAR_COUNT).fill(MIN_H);
  const target = new Array(BAR_COUNT).fill(MIN_H);
  const phase = Array.from({ length: BAR_COUNT }, () => Math.random() * Math.PI * 2);

  // --- public API for Python ---
  window.setState = (s) => {
    if (s === 'idle' || s === 'listening' || s === 'thinking' || s === 'speaking') {
      state = s;
    }
  };
  window.feedAmplitude = (level) => {
    if (typeof level !== 'number' || isNaN(level)) return;
    if (level < 0) level = 0;
    if (level > 1) level = 1;
    amp = Math.max(amp, level);  // peak hold; decays in tick
  };

  // --- per-state target update ---
  function updateTargets(now) {
    if (state === 'listening') {
      // amplitude-driven with per-bar jitter; center bar gets a boost
      for (let i = 0; i < BAR_COUNT; i++) {
        const dist = Math.abs(i - (BAR_COUNT - 1) / 2);
        const weight = 1.0 - 0.18 * dist;
        const jitter = 0.85 + Math.random() * 0.30;
        const v = Math.min(1, amp * weight * jitter);
        target[i] = MIN_H + (MAX_H - MIN_H) * v;
      }
    } else if (state === 'thinking') {
      // gentle sinusoidal pulse
      const t = now / 1000;
      for (let i = 0; i < BAR_COUNT; i++) {
        const v = 0.32 + 0.18 * Math.sin(t * 2.2 + phase[i]);
        target[i] = MIN_H + (MAX_H - MIN_H) * v;
      }
    } else if (state === 'speaking') {
      // random bouncy
      for (let i = 0; i < BAR_COUNT; i++) {
        const v = 0.42 + Math.random() * 0.5;
        target[i] = MIN_H + (MAX_H - MIN_H) * v;
      }
    } else {
      // idle
      for (let i = 0; i < BAR_COUNT; i++) target[i] = MIN_H;
    }
  }

  // --- drawing: pill = body rect + two arcs ---
  function drawPill(cx, cy, w, h) {
    const r = w / 2;
    const halfH = h / 2;
    ctx.beginPath();
    // body
    ctx.rect(cx - r, cy - halfH, w, h);
    // top cap (full circle that overlaps the rect)
    ctx.moveTo(cx + r, cy - halfH);
    ctx.arc(cx, cy - halfH, r, 0, Math.PI * 2);
    // bottom cap
    ctx.moveTo(cx + r, cy + halfH);
    ctx.arc(cx, cy + halfH, r, 0, Math.PI * 2);
    ctx.fillStyle = BAR_COLOR;
    ctx.fill();
  }

  function render() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);
    const total = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_SPACING;
    const startX = (w - total) / 2 + BAR_WIDTH / 2;
    const cy = h / 2;
    for (let i = 0; i < BAR_COUNT; i++) {
      const cx = startX + i * (BAR_WIDTH + BAR_SPACING);
      drawPill(cx, cy, BAR_WIDTH, cur[i]);
    }
  }

  function tick(now) {
    updateTargets(now);
    for (let i = 0; i < BAR_COUNT; i++) {
      cur[i] += (target[i] - cur[i]) * SMOOTH;
    }
    amp *= AMP_DECAY;
    render();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // Debug helper — let dev cycle states from the browser console
  window.__javisDemo = () => {
    const seq = ['listening', 'thinking', 'speaking', 'listening', 'idle'];
    let i = 0;
    setInterval(() => {
      window.setState(seq[i % seq.length]);
      i++;
    }, 3000);
    setInterval(() => {
      if (state === 'listening') window.feedAmplitude(0.3 + 0.6 * Math.random() * Math.random());
    }, 50);
  };
})();
