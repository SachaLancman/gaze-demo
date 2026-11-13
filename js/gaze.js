/* Maison Lunaire — cursor-follow gaze
 *
 * Angle from the head on screen to the cursor picks the gaze direction.
 * Distance from the head picks how far he looks (inner vs outer poses).
 * Cursor on the face → rest frame, looking at the viewer.
 * Frames are pre-baked WebPs drawn to a canvas; nothing is "played".
 * Glide takes the shorter way around the loop (preferring covered arcs).
 */
(() => {
  "use strict";

  const IDLE_MS = 2500;
  const GLIDE_TAU = 0.12; // seconds; frame-rate independent
  const AUTO_PERIOD = 10; // seconds for a full idle sweep of the covered arc
  const FACE_PAD = 1.05; // rest-zone scale on the face ellipse
  const CONCURRENCY = 8;

  const canvas = document.getElementById("stage");
  const poster = document.getElementById("poster");
  const loader = document.getElementById("loader");
  const hint = document.getElementById("hint");
  const ctx = canvas.getContext("2d", { alpha: false });

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const snap = new URLSearchParams(location.search).has("snap");

  let meta = null;
  let frames = [];
  let images = [];
  let loaded = new Uint8Array(0);
  let loadedCount = 0;

  let curAngle = 0;
  let curR = 0;
  let tgtAngle = 0;
  let tgtR = 0;
  let currentIndex = -1;
  let lastTs = 0;
  let lastMoveAt = 0;
  let idle = false;
  let idleT = 0;
  let hintHidden = false;
  let running = false;
  let orientationOn = false;
  let pointerX = 0.5;
  let pointerY = 0.5;
  let hasPointer = false;
  let overHero = true;

  // Covered horseshoe of the outer clip, used for idle look-around
  // and to prefer a glide that stays on footage.
  let coveredArc = null; // { start, end } in radians, going CCW (increasing) the long covered way
  let outerMedR = 15;

  function shortestDelta(a, b) {
    let d = b - a;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    return d;
  }

  function normAngle(a) {
    while (a > Math.PI) a -= Math.PI * 2;
    while (a < -Math.PI) a += Math.PI * 2;
    return a;
  }

  function loadImage(i) {
    if (loaded[i] === 1) return Promise.resolve(images[i]);
    if (loaded[i] === 2) return images[i]._p;
    loaded[i] = 2;
    const p = new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        images[i] = img;
        loaded[i] = 1;
        loadedCount++;
        resolve(img);
      };
      img.onerror = () => {
        loaded[i] = 0;
        reject(new Error("fail " + frames[i].file));
      };
      img.src = frames[i].file;
    });
    images[i] = { _p: p };
    return p;
  }

  async function loadMany(indices) {
    const q = indices.filter((i) => loaded[i] !== 1);
    let k = 0;
    async function worker() {
      while (k < q.length) {
        const i = q[k++];
        try {
          await loadImage(i);
        } catch (_) {
          /* skip */
        }
      }
    }
    const n = Math.min(CONCURRENCY, q.length);
    await Promise.all(Array.from({ length: n }, worker));
  }

  function farthestPointSample(n) {
    // Always include rest and a spread across (yaw, pitch).
    const rest = meta.restIndex | 0;
    const pts = frames.map((f, i) => ({ i, y: f.yaw, p: f.pitch }));
    const picked = [rest];
    const used = new Set([rest]);
    while (picked.length < Math.min(n, pts.length)) {
      let best = -1,
        bestD = -1;
      for (const q of pts) {
        if (used.has(q.i)) continue;
        let minD = Infinity;
        for (const pi of picked) {
          const f = frames[pi];
          const d = (q.y - f.yaw) ** 2 + (q.p - f.pitch) ** 2;
          if (d < minD) minD = d;
        }
        if (minD > bestD) {
          bestD = minD;
          best = q.i;
        }
      }
      if (best < 0) break;
      picked.push(best);
      used.add(best);
    }
    return picked;
  }

  function buildCoveredArc() {
    const outer = frames.filter((f) => f.src === "outer" && f.radius > 6);
    const rads = outer
      .map((f) => (f.angle * Math.PI) / 180)
      .sort((a, b) => a - b);
    if (rads.length < 4) {
      coveredArc = { start: -Math.PI, end: Math.PI, span: Math.PI * 2 };
      return;
    }
    // Largest gap between successive angles (plus wrap) is the uncovered south opening.
    let maxGap = 0,
      gapAfter = 0;
    for (let i = 0; i < rads.length; i++) {
      const a = rads[i];
      const b = i + 1 < rads.length ? rads[i + 1] : rads[0] + Math.PI * 2;
      const g = b - a;
      if (g > maxGap) {
        maxGap = g;
        gapAfter = i;
      }
    }
    const start = normAngle(
      (gapAfter + 1 < rads.length ? rads[gapAfter + 1] : rads[0])
    );
    const end = rads[gapAfter];
    let span = shortestDelta(start, end);
    // We want the LONG way (the covered horseshoe), so if span is the short (gap), flip.
    if (Math.abs(span) < maxGap * 0.5 || Math.abs(span) < Math.PI * 0.4) {
      span = span > 0 ? span - Math.PI * 2 : span + Math.PI * 2;
    }
    // Force travelling the covered direction: from start (just after the gap) along increasing
    // or decreasing to end, with |span| ≈ 2π - maxGap.
    const want = Math.PI * 2 - maxGap;
    if (Math.abs(span) < want * 0.7) {
      span = span >= 0 ? want : -want;
    }
    coveredArc = { start, end, span, gap: maxGap };
    const rs = outer.map((f) => f.radius);
    rs.sort((a, b) => a - b);
    outerMedR = rs[(rs.length / 2) | 0];
  }

  function arcPassesGap(from, delta) {
    if (!coveredArc || coveredArc.gap < 0.4) return false;
    // Sample a few points along the proposed arc; if many have no nearby frame, it's the gap.
    const steps = 6;
    let misses = 0;
    for (let s = 1; s <= steps; s++) {
      const a = from + (delta * s) / (steps + 1);
      const yaw = outerMedR * Math.cos(a);
      const pitch = outerMedR * Math.sin(a);
      const { dist } = nearestInfo(yaw, pitch, false);
      if (dist > 6) misses++;
    }
    return misses >= 3;
  }

  function preferredDelta(from, to) {
    const d = shortestDelta(from, to);
    const other = d > 0 ? d - Math.PI * 2 : d + Math.PI * 2;
    const shortBad = arcPassesGap(from, d);
    const longBad = arcPassesGap(from, other);
    if (shortBad && !longBad) return other;
    return d;
  }

  function nearestInfo(yaw, pitch, onlyLoaded) {
    let best = meta.restIndex | 0;
    let bestD = Infinity;
    for (let i = 0; i < frames.length; i++) {
      if (onlyLoaded && loaded[i] !== 1) continue;
      const f = frames[i];
      const d = (f.yaw - yaw) ** 2 + (f.pitch - pitch) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return { i: best, dist: Math.sqrt(bestD) };
  }

  function nearest(yaw, pitch) {
    // Prefer a loaded frame; fall back to any if the closest is already loaded or almost as close.
    const any = nearestInfo(yaw, pitch, false);
    if (loaded[any.i] === 1) return any.i;
    const got = nearestInfo(yaw, pitch, true);
    return got.i;
  }

  function placement() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = canvas.width / dpr;
    const ch = canvas.height / dpr;
    const vw = meta.width;
    const vh = meta.height;
    const scale = Math.max(cw / vw, ch / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    let ox = (cw - dw) / 2;
    let oy = (ch - dh) / 2;
    // Keep the face near 36% of the viewport height (matches CSS object-position).
    const face = meta.face;
    const faceY = oy + face.cy * dh;
    const want = ch * 0.36;
    oy += want - faceY;
    oy = Math.max(Math.min(0, oy), ch - dh);
    ox = Math.max(Math.min(0, ox), cw - dw);
    return { cw, ch, vw, vh, scale, dw, dh, ox, oy, face };
  }

  function headOnScreen() {
    const p = placement();
    return {
      x: p.ox + p.face.cx * p.dw,
      y: p.oy + p.face.cy * p.dh,
      rx: p.face.rx * p.dw,
      ry: p.face.ry * p.dh,
      p,
    };
  }

  function distToViewportEdge(x, y, angle, cw, ch) {
    // Ray from (x,y) at `angle` (0=right, π/2=up) to the viewport edge.
    const dx = Math.cos(angle);
    const dy = -Math.sin(angle); // screen y is down
    let t = Infinity;
    if (dx > 1e-6) t = Math.min(t, (cw - x) / dx);
    else if (dx < -1e-6) t = Math.min(t, (0 - x) / dx);
    if (dy > 1e-6) t = Math.min(t, (ch - y) / dy);
    else if (dy < -1e-6) t = Math.min(t, (0 - y) / dy);
    return Number.isFinite(t) ? Math.max(1, t) : Math.hypot(cw, ch) / 2;
  }

  function cursorToTarget(cx, cy) {
    const h = headOnScreen();
    const dx = cx - h.x;
    const dy = cy - h.y;
    const dist = Math.hypot(dx, dy);
    const angle = Math.atan2(-dy, dx); // 0=right, π/2=up
    // Rest zone: a slightly padded oval, but not so tall that
    // the forehead reaches the top of the screen (which would
    // leave almost no travel for an upward look).
    // Tight vis-age oval (eyes, nose, cheeks) — hair and collar do not count
    // as "on his face", so there is room above the head to look fully up.
    const rx = h.rx * 0.95;
    const ry = h.ry * 0.52;
    const nx = dx / rx;
    const ny = dy / ry;
    if (nx * nx + ny * ny <= 1) {
      return { angle, r: 0, yaw: 0, pitch: 0, rest: true };
    }
    const faceDist = Math.hypot(rx * Math.cos(angle), ry * Math.sin(angle));
    const maxDist = distToViewportEdge(h.x, h.y, angle, h.p.cw, h.p.ch);
    const t = Math.min(1, Math.max(0, (dist - faceDist) / Math.max(1, maxDist - faceDist)));
    // Screen edge maps to max observed angles (elliptical in yaw/pitch).
    const yaw = t * meta.maxAbsYaw * Math.cos(angle);
    const pitch = t * meta.maxAbsPitch * Math.sin(angle);
    const r = Math.hypot(yaw, pitch);
    return { angle, r, yaw, pitch, rest: false };
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (currentIndex >= 0) paint(currentIndex);
  }

  function paint(i) {
    if (loaded[i] !== 1) return;
    const img = images[i];
    const p = placement();
    ctx.fillStyle = "#080706";
    ctx.fillRect(0, 0, p.cw, p.ch);
    ctx.drawImage(img, p.ox, p.oy, p.dw, p.dh);
    currentIndex = i;
    if (poster && poster.style.opacity !== "0") {
      poster.style.opacity = "0";
      poster.style.transition = "opacity 0.4s linear";
    }
  }

  function applyTarget(tgt, instant) {
    tgtAngle = tgt.angle;
    tgtR = tgt.r;
    if (tgt.rest) tgtR = 0;
    if (instant || snap) {
      curAngle = tgtAngle;
      curR = tgtR;
      const yaw = curR * Math.cos(curAngle);
      const pitch = curR * Math.sin(curAngle);
      const i = tgt.rest || curR < 0.35 ? meta.restIndex | 0 : nearest(yaw, pitch);
      paint(i);
    }
  }

  function setFromPointer(clientX, clientY, instant) {
    pointerX = clientX;
    pointerY = clientY;
    hasPointer = true;
    lastMoveAt = performance.now();
    if (idle) idle = false;
    const hero = document.getElementById("hero");
    const r = hero.getBoundingClientRect();
    overHero = clientY >= r.top && clientY <= r.bottom;
    if (!overHero) {
      applyTarget({ angle: curAngle, r: 0, rest: true }, instant);
      return;
    }
    const tgt = cursorToTarget(clientX, clientY);
    applyTarget(tgt, instant);
    if (!hintHidden && !tgt.rest) {
      hintHidden = true;
      if (hint) hint.classList.add("is-hidden");
    }
  }

  function tick(ts) {
    if (!running) return;
    requestAnimationFrame(tick);
    const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0.016;
    lastTs = ts;

    if (reduced) return;

    // Idle / phone auto look-around
    const now = ts;
    const shouldIdle =
      orientationOn
        ? false
        : (coarsePointer && !hasPointer) ||
          (hasPointer && now - lastMoveAt > IDLE_MS) ||
          (coarsePointer && now - lastMoveAt > IDLE_MS);
    if (shouldIdle && !idle) {
      idle = true;
      idleT = 0;
      // Start the sweep from the current angle, travelling the covered horseshoe.
    }
    if (idle && coveredArc) {
      idleT += dt;
      const frac = (idleT / AUTO_PERIOD) % 1;
      // Ease along the covered span.
      const a = coveredArc.start + coveredArc.span * frac;
      tgtAngle = normAngle(a);
      tgtR = outerMedR;
    }

    const k = snap ? 1 : 1 - Math.exp(-dt / GLIDE_TAU);
    const dAng = preferredDelta(curAngle, tgtAngle);
    curAngle = normAngle(curAngle + dAng * k);
    curR += (tgtR - curR) * k;

    const restI = meta.restIndex | 0;
    let i;
    if (curR < 0.35) {
      i = restI;
    } else {
      const yaw = curR * Math.cos(curAngle);
      const pitch = curR * Math.sin(curAngle);
      i = nearest(yaw, pitch);
    }
    if (i !== currentIndex) paint(i);
  }

  function bind() {
    window.addEventListener(
      "pointermove",
      (e) => {
        if (e.pointerType === "touch" && e.buttons === 0 && e.pressure === 0) {
          // Some browsers fire a lingering move; still count it.
        }
        setFromPointer(e.clientX, e.clientY, false);
      },
      { passive: true }
    );
    window.addEventListener(
      "pointerdown",
      (e) => setFromPointer(e.clientX, e.clientY, false),
      { passive: true }
    );
    window.addEventListener(
      "pointerleave",
      () => {
        hasPointer = false;
        lastMoveAt = performance.now() - IDLE_MS; // allow idle soon
      },
      { passive: true }
    );

    // Device orientation, if the user grants it (phones).
    const enableOrient = () => {
      const handler = (e) => {
        if (e.gamma == null || e.beta == null) return;
        orientationOn = true;
        lastMoveAt = performance.now();
        idle = false;
        // gamma: left(-) / right(+), roughly -45..45 useful
        // beta: 90 is upright; less = tilt back (look up), more = tilt forward (look down)
        const yawN = Math.max(-1, Math.min(1, e.gamma / 28));
        const pitchN = Math.max(-1, Math.min(1, (90 - e.beta) / 28));
        const angle = Math.atan2(pitchN, yawN);
        const mag = Math.min(1, Math.hypot(yawN, pitchN));
        const r = mag * Math.hypot(meta.maxAbsYaw, meta.maxAbsPitch) * 0.85;
        applyTarget({ angle, r, rest: mag < 0.12 }, false);
      };
      window.addEventListener("deviceorientation", handler);
    };
    const maybeAsk = () => {
      const DOE = window.DeviceOrientationEvent;
      if (!DOE) return;
      if (typeof DOE.requestPermission === "function") {
        DOE.requestPermission()
          .then((s) => {
            if (s === "granted") enableOrient();
          })
          .catch(() => {});
      } else if (coarsePointer) {
        enableOrient();
      }
    };
    window.addEventListener("pointerdown", maybeAsk, { once: true });

    const form = document.getElementById("viewingForm");
    if (form) {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const thanks = document.getElementById("thanks");
        if (thanks) thanks.hidden = false;
        form.reset();
      });
    }

    // Hide hint on phones after a moment; replace copy.
    if (hint && coarsePointer) {
      hint.textContent = "Tilt or touch  ·  he will look around";
    }
  }

  // Test / screenshot hooks
  window.__gazeGoto = function (nx, ny, instant = true) {
    // nx, ny: 0..1 in viewport (0.5, 0.5 is center, not the head).
    const x = nx * window.innerWidth;
    const y = ny * window.innerHeight;
    setFromPointer(x, y, instant);
    // Force a paint at the exact target (skip leftover glide).
    if (instant) {
      const tgt = cursorToTarget(x, y);
      curAngle = tgt.angle;
      curR = tgt.r;
      const i =
        tgt.rest || curR < 0.35
          ? meta.restIndex | 0
          : nearest(curR * Math.cos(curAngle), curR * Math.sin(curAngle));
      paint(i);
    }
    return window.__gazeState();
  };
  window.__gazeLook = function (dir) {
    // Snap to a cardinal direction from the head, near the viewport edge.
    const h = headOnScreen();
    const map = {
      left: [0.04, h.y / window.innerHeight],
      right: [0.96, h.y / window.innerHeight],
      up: [h.x / window.innerWidth, 0.02],
      down: [h.x / window.innerWidth, 0.92],
      rest: [h.x / window.innerWidth, h.y / window.innerHeight],
    };
    const [nx, ny] = map[dir] || map.rest;
    return window.__gazeGoto(nx, ny, true);
  };
  window.__gazeState = function () {
    const f = frames[currentIndex] || {};
    return {
      index: currentIndex,
      src: f.src,
      file: f.file,
      yaw: f.yaw,
      pitch: f.pitch,
      angle: (curAngle * 180) / Math.PI,
      r: curR,
      loaded: loadedCount,
      total: frames.length,
    };
  };
  window.__gazeReady = function () {
    return loadedCount >= Math.min(36, frames.length) && currentIndex >= 0;
  };

  async function init() {
    resize();
    window.addEventListener("resize", resize);

    meta = await (await fetch("frames.json")).json();
    frames = meta.frames;
    images = new Array(frames.length);
    loaded = new Uint8Array(frames.length);
    buildCoveredArc();

    const restI = meta.restIndex | 0;
    await loadImage(restI);
    paint(restI);

    if (reduced) {
      hideLoader();
      return;
    }

    const coarse = farthestPointSample(48);
    await loadMany(coarse);
    hideLoader();
    bind();
    running = true;
    lastMoveAt = performance.now();
    requestAnimationFrame(tick);

    // Background-load the rest, lazy, without blocking.
    loadMany(frames.map((_, i) => i)).catch(() => {});
  }

  function hideLoader() {
    if (loader) loader.classList.add("is-done");
  }

  init().catch((err) => {
    console.error(err);
    hideLoader();
  });
})();
