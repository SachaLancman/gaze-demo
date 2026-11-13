#!/usr/bin/env python3
"""Extract, pose-estimate, filter, and pack gaze frames.

Discovers every .mp4 in media/ except gaze-loop.mp4 (the closed-eye v1).
New clips (e.g. media/gaze-mid.mp4) are picked up with no code changes.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path("/workspace/sites/gaze-demo")
MEDIA = ROOT / "media"
FRAMES_OUT = ROOT / "frames"
WORK = Path("/tmp/gaze-work")
MODEL = Path("/tmp/face_landmarker.task")
SKIP_CLIPS = {"gaze-loop.mp4"}  # v1, eyes closed
# Friendly source labels; unknown stems keep their stem name.
ALIAS = {
    "gaze-loop-v2": "outer",
    "gaze-inner": "inner",
}

# MediaPipe Face Mesh landmark indices
# Subject's right eye (left side of image): outer, top, top-in, inner, bot-in, bot
REYE = [33, 160, 158, 133, 153, 144]
LEYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# solvePnP model (mm), subject's left = +X in some conventions — we use
# image-left = subject's right = +X_image, but the 3D model is subject-centric:
# subject's left eye at -X (standard OpenCV head-pose model).
# Indices: nose, chin, subject-left-eye-outer (263), subject-right-eye-outer (33),
# subject-left-mouth (291), subject-right-mouth (61)
PNP_IDX = [1, 152, 263, 33, 291, 61]
PNP_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def ear(lms, idx):
    p = np.array([[lms[i].x, lms[i].y] for i in idx], dtype=np.float64)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    if c < 1e-9:
        return 0.0
    return float((a + b) / (2.0 * c))


def euler_from_R(R):
    """Return pitch (x), yaw (y), roll (z) in degrees from a 3x3 rotation."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def euler_from_matrix_alt(R):
    """Alternate Tait-Bryan (pitch, yaw, roll) used widely with MediaPipe."""
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def discover_clips():
    clips = []
    for p in sorted(MEDIA.glob("*.mp4")):
        if p.name in SKIP_CLIPS:
            print(f"skip {p.name} (excluded)")
            continue
        stem = p.stem
        src = ALIAS.get(stem, stem)
        clips.append({"path": p, "stem": stem, "src": src})
    if not clips:
        sys.exit("no clips found in media/")
    return clips


def extract(clip, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.glob("f_*.png"))
    if len(existing) >= 100:
        print(f"reuse {len(existing)} pngs in {dest}")
        return
    for f in dest.glob("*"):
        f.unlink()
    cmd = [
        "ffmpeg", "-y", "-i", str(clip["path"]),
        "-vf", "fps=24,scale=848:-2",
        "-q:v", "2",
        str(dest / "f_%04d.png"),
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.check_call(cmd)
    print(f"extracted {len(list(dest.glob('f_*.png')))} from {clip['path'].name}")


def make_landmarker():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def blink_from_blendshapes(result):
    left = right = None
    if not result.face_blendshapes:
        return None, None
    for cat in result.face_blendshapes[0]:
        if cat.category_name == "eyeBlinkLeft":
            left = cat.score
        elif cat.category_name == "eyeBlinkRight":
            right = cat.score
    return left, right


def detect_one(landmarker, bgr, mp_mod):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    h, w = bgr.shape[:2]

    # Transformation-matrix Euler
    mat_pitch = mat_yaw = mat_roll = None
    if result.facial_transformation_matrixes:
        M = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
        R = M[:3, :3]
        mat_pitch, mat_yaw, mat_roll = euler_from_matrix_alt(R)

    # solvePnP Euler
    pts_2d = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in PNP_IDX], dtype=np.float64
    )
    cam = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64
    )
    pnp_pitch = pnp_yaw = pnp_roll = pnp_err = None
    ok, rvec, tvec = cv2.solvePnP(PNP_3D, pts_2d, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        # OpenCV: +yaw often means subject turning to their left (viewer's right)
        pnp_pitch, pnp_yaw, pnp_roll = euler_from_R(R)
        proj, _ = cv2.projectPoints(PNP_3D, rvec, tvec, cam, None)
        pnp_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts_2d, axis=1)))

    # Nose offset relative to eye midpoint — viewer-perspective proxy
    re = np.array([lms[33].x, lms[33].y])
    le = np.array([lms[263].x, lms[263].y])
    mid = (re + le) / 2
    nose = np.array([lms[NOSE_TIP].x, lms[NOSE_TIP].y])
    iod = np.linalg.norm(le - re) + 1e-9
    # +nx = nose to the right of eye mid (looking right, viewer's POV)
    # +ny = nose above eye mid (looking up); image y grows down so flip
    nose_x = float((nose[0] - mid[0]) / iod)
    nose_y = float(-(nose[1] - mid[1]) / iod)

    # Face center / radius in normalized frame coords (for JS rest-zone)
    oval = np.array([[lms[i].x, lms[i].y] for i in FACE_OVAL])
    face_cx = float(oval[:, 0].mean())
    face_cy = float(oval[:, 1].mean())
    face_rx = float((oval[:, 0].max() - oval[:, 0].min()) / 2)
    face_ry = float((oval[:, 1].max() - oval[:, 1].min()) / 2)

    e_l = ear(lms, LEYE)
    e_r = ear(lms, REYE)
    e = (e_l + e_r) / 2
    blink_l, blink_r = blink_from_blendshapes(result)

    # Warp heuristic: eye-line roll vs head roll, oval aspect, pnp error
    eye_dx = le[0] - re[0]
    eye_dy = le[1] - re[1]
    eye_roll = math.degrees(math.atan2(eye_dy, eye_dx))

    return {
        "mat_pitch": mat_pitch,
        "mat_yaw": mat_yaw,
        "mat_roll": mat_roll,
        "pnp_pitch": pnp_pitch,
        "pnp_yaw": pnp_yaw,
        "pnp_roll": pnp_roll,
        "pnp_err": pnp_err,
        "nose_x": nose_x,
        "nose_y": nose_y,
        "face_cx": face_cx,
        "face_cy": face_cy,
        "face_rx": face_rx,
        "face_ry": face_ry,
        "ear": e,
        "ear_l": e_l,
        "ear_r": e_r,
        "blink_l": blink_l,
        "blink_r": blink_r,
        "eye_roll": eye_roll,
        "iod": float(iod),
    }


def measure_clip(landmarker, mp_mod, clip, png_dir: Path):
    files = sorted(png_dir.glob("f_*.png"))
    rows = []
    misses = 0
    for i, fp in enumerate(files):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            misses += 1
            continue
        det = detect_one(landmarker, bgr, mp_mod)
        if det is None:
            misses += 1
            print(f"  no face {clip['src']} {fp.name}")
            continue
        det.update(
            {
                "src": clip["src"],
                "stem": clip["stem"],
                "png": str(fp),
                "i_src": i,
                "t": i / 24.0,
                "w": bgr.shape[1],
                "h": bgr.shape[0],
            }
        )
        rows.append(det)
        if (i + 1) % 50 == 0:
            print(f"  {clip['src']}: {i+1}/{len(files)}")
    print(f"  {clip['src']}: measured {len(rows)}, misses {misses}")
    return rows


def choose_convention(rows):
    """Pick Euler signs so +yaw is viewer's right and +pitch is up,
    matching the known outer-clip timeline (right~1s, up~2s, left~3s, down~4s).
    """
    outer = [r for r in rows if r["src"] == "outer"]
    if not outer:
        # No labelled outer clip: correlate with nose offset.
        def corr(key_a, key_b, sign=1):
            a = np.array([r[key_a] for r in rows if r[key_a] is not None])
            b = np.array([r[key_b] for r in rows if r[key_a] is not None]) * sign
            if len(a) < 5:
                return 0
            return float(np.corrcoef(a, b)[0, 1])

        use = "mat"
        yaw_sign = 1 if corr("mat_yaw", "nose_x") >= 0 else -1
        pitch_sign = 1 if corr("mat_pitch", "nose_y") >= 0 else -1
        return use, yaw_sign, pitch_sign

    def at(t, tol=0.2):
        cand = [r for r in outer if abs(r["t"] - t) <= tol]
        return cand[len(cand) // 2] if cand else None

    samples = {
        "right": at(1.0),
        "up": at(2.0),
        "left": at(3.0),
        "down": at(4.0),
        "center": at(0.0) or at(8.0),
    }
    print("\nSanity — raw angles at known timestamps:")
    for name, r in samples.items():
        if not r:
            print(f"  {name}: missing")
            continue
        print(
            f"  {name} t={r['t']:.2f}  mat yaw/pitch/roll={r['mat_yaw']:.1f}/{r['mat_pitch']:.1f}/{r['mat_roll']:.1f}"
            f"  pnp={r['pnp_yaw']:.1f}/{r['pnp_pitch']:.1f}/{r['pnp_roll']:.1f}"
            f"  nose={r['nose_x']:.3f}/{r['nose_y']:.3f}"
            f"  ear={r['ear']:.3f} blink={r['blink_l']},{r['blink_r']}"
        )

    # PnP/matrix Euler proved gimbal-unstable on this footage (yaw flips to ±170°,
    # pitch swallowed the left/right motion). Nose offset is viewer-aligned and
    # stable: +nose_x = looking right, +nose_y (recentered) = looking up.
    print("Forcing nose-offset convention (yaw_sign=1, pitch_sign=1).")
    return "nose", 1, 1

    # Score each (source, yaw_sign, pitch_sign)
    best = ("mat", 1, 1, -1e9)
    for use in ("mat", "pnp", "nose"):
        for ys in (1, -1):
            for ps in (1, -1):
                def yp(r):
                    if use == "nose":
                        return ys * r["nose_x"], ps * r["nose_y"]
                    return ys * r[f"{use}_yaw"], ps * r[f"{use}_pitch"]

                score = 0.0
                rgt, up, lft, dn, ctr = (
                    samples["right"],
                    samples["up"],
                    samples["left"],
                    samples["down"],
                    samples["center"],
                )
                if rgt and lft:
                    y_r, _ = yp(rgt)
                    y_l, _ = yp(lft)
                    if y_r > 0:
                        score += 2
                    if y_l < 0:
                        score += 2
                    score += (y_r - y_l)  # magnitude bonus if signs right
                if up and dn:
                    _, p_u = yp(up)
                    _, p_d = yp(dn)
                    if p_u > 0:
                        score += 2
                    if p_d < 0:
                        score += 2
                    score += (p_u - p_d)
                if ctr:
                    y, p = yp(ctr)
                    score -= 0.15 * (abs(y) + abs(p))
                if score > best[3]:
                    best = (use, ys, ps, score)
    use, ys, ps, sc = best
    print(f"\nChosen convention: {use} yaw_sign={ys} pitch_sign={ps} score={sc:.2f}")
    return use, ys, ps


def apply_convention(rows, use, ys, ps):
    for r in rows:
        if use == "nose":
            # Scale nose offsets into rough degrees so JS math is uniform.
            r["yaw_raw"] = ys * r["nose_x"] * 35.0
            r["pitch_raw"] = ps * r["nose_y"] * 35.0
            r["roll"] = r["eye_roll"]
        else:
            r["yaw_raw"] = ys * r[f"{use}_yaw"]
            r["pitch_raw"] = ps * r[f"{use}_pitch"]
            r["roll"] = r[f"{use}_roll"]


def origin_and_recenter(rows):
    """Subtract a robust looking-at-camera origin."""
    # Prefer outer frames near t=0 and t=8, plus inner frames with smallest radius
    # after a first-pass median.
    my = np.median([r["yaw_raw"] for r in rows])
    mp_ = np.median([r["pitch_raw"] for r in rows])
    # First-pass radius around median
    for r in rows:
        r["_r0"] = math.hypot(r["yaw_raw"] - my, r["pitch_raw"] - mp_)
    inner = [r for r in rows if r["src"] == "inner"]
    outer = [r for r in rows if r["src"] == "outer"]
    seeds = []
    if outer:
        # start and end of outer loop should be near center
        seeds += [r for r in outer if r["t"] < 0.25 or r["t"] > 7.6]
    if inner:
        # smallest 15% of inner radii — the circle's interior, near its center
        inn = sorted(inner, key=lambda r: r["_r0"])
        seeds += inn[: max(8, len(inn) // 7)]
    if not seeds:
        seeds = sorted(rows, key=lambda r: r["_r0"])[: max(8, len(rows) // 10)]
    oy = float(np.median([r["yaw_raw"] for r in seeds]))
    op = float(np.median([r["pitch_raw"] for r in seeds]))
    print(f"Origin (look-at-camera): yaw={oy:.2f} pitch={op:.2f}  (n_seeds={len(seeds)})")
    for r in rows:
        r["yaw"] = r["yaw_raw"] - oy
        r["pitch"] = r["pitch_raw"] - op
        r["radius"] = math.hypot(r["yaw"], r["pitch"])
        r["angle"] = math.degrees(math.atan2(r["pitch"], r["yaw"]))  # 0=right, 90=up
    return oy, op


def filter_rows(rows):
    """Drop closed eyes and warped / unreliable faces."""
    ears = np.array([r["ear"] for r in rows])
    ear_med = float(np.median(ears))
    ear_thr = max(0.14, ear_med * 0.62)
    blinks = [((r["blink_l"] or 0) + (r["blink_r"] or 0)) / 2 for r in rows]
    blink_thr = 0.42
    pnp_errs = [r["pnp_err"] for r in rows if r["pnp_err"] is not None]
    pnp_thr = float(np.percentile(pnp_errs, 92)) if pnp_errs else 1e9
    rolls = np.array([abs(r["roll"] or 0) for r in rows])
    roll_thr = max(28.0, float(np.percentile(rolls, 97)))
    iods = np.array([r["iod"] for r in rows])
    iod_lo, iod_hi = np.percentile(iods, [1, 99])
    iod_lo = min(iod_lo, np.median(iods) * 0.7)
    iod_hi = max(iod_hi, np.median(iods) * 1.35)

    kept, dropped = [], []
    for r in rows:
        reasons = []
        blink = ((r["blink_l"] or 0) + (r["blink_r"] or 0)) / 2
        if r["ear"] < ear_thr:
            reasons.append(f"ear={r['ear']:.3f}<{ear_thr:.3f}")
        if blink > blink_thr:
            reasons.append(f"blink={blink:.2f}")
        if r["pnp_err"] is not None and r["pnp_err"] > pnp_thr * 1.15:
            reasons.append(f"pnp_err={r['pnp_err']:.1f}")
        if abs(r["roll"] or 0) > roll_thr:
            reasons.append(f"roll={r['roll']:.1f}")
        if r["iod"] < iod_lo * 0.85 or r["iod"] > iod_hi * 1.15:
            reasons.append(f"iod={r['iod']:.3f}")
        # Asymmetric collapse of one eye (warp / wink)
        if min(r["ear_l"], r["ear_r"]) < 0.10 and max(r["ear_l"], r["ear_r"]) > 0.22:
            reasons.append("asymmetric_eye")
        if reasons:
            r["drop"] = reasons
            dropped.append(r)
        else:
            kept.append(r)
    print(f"\nFilter: ear_thr={ear_thr:.3f} (med={ear_med:.3f}) blink_thr={blink_thr} pnp_thr={pnp_thr:.2f} roll_thr={roll_thr:.1f}")
    print(f"  kept {len(kept)} / {len(rows)}   dropped {len(dropped)}")
    by = defaultdict(list)
    for r in dropped:
        by[r["src"]].append(r)
    for src, rs in by.items():
        ts = ", ".join(f"{x['t']:.2f}s({','.join(x['drop'])})" for x in rs[:8])
        more = f" …+{len(rs)-8}" if len(rs) > 8 else ""
        print(f"  dropped {src}: {len(rs)}  e.g. {ts}{more}")
    return kept, dropped


def light_dedup(rows, deg=0.55):
    """Drop near-duplicate poses, keeping the opener-eyed / sharper frame."""
    # Sort by quality then greedily keep
    def quality(r):
        blink = ((r["blink_l"] or 0) + (r["blink_r"] or 0)) / 2
        return r["ear"] - blink - 0.01 * abs(r["roll"] or 0)

    ordered = sorted(rows, key=quality, reverse=True)
    kept = []
    for r in ordered:
        if any(math.hypot(r["yaw"] - k["yaw"], r["pitch"] - k["pitch"]) < deg for k in kept):
            r["dedup"] = True
            continue
        kept.append(r)
    kept.sort(key=lambda r: (r["src"], r["i_src"]))
    print(f"Dedup @{deg}°: {len(rows)} → {len(kept)}")
    return kept


def encode_webp(rows, quality=62):
    FRAMES_OUT.mkdir(parents=True, exist_ok=True)
    # wipe old webps (keep nothing stale) but not our temp previews if any already gone
    for p in FRAMES_OUT.glob("*.webp"):
        p.unlink()
    for p in FRAMES_OUT.glob("*strip*"):
        p.unlink()
    for p in FRAMES_OUT.glob("portrait*"):
        p.unlink()

    # Also encode a rest-frame from portrait.jpg, resized to 848x480 cover
    # (done by caller). Here just the clip frames.
    for n, r in enumerate(rows):
        out_name = f"{r['src']}_{r['i_src']:04d}.webp"
        out = FRAMES_OUT / out_name
        im = Image.open(r["png"]).convert("RGB")
        if im.size != (848, 480):
            im = im.resize((848, 480), Image.Resampling.LANCZOS)
        im.save(out, "WEBP", quality=quality, method=6)
        r["file"] = f"frames/{out_name}"
        if (n + 1) % 80 == 0:
            print(f"  encoded {n+1}/{len(rows)}")
    total = sum(p.stat().st_size for p in FRAMES_OUT.glob("*.webp"))
    print(f"Encoded {len(rows)} webp q={quality}: {total/1e6:.2f} MB")
    return total


def encode_rest_portrait(rows):
    """Add a high-quality rest frame from portrait.jpg, posed at (0,0)."""
    src = MEDIA / "portrait.jpg"
    if not src.exists():
        return None
    im = Image.open(src).convert("RGB")
    # Cover-fit into 848x480 (portrait is 1280x720, same aspect 16:9)
    tw, th = 848, 480
    scale = max(tw / im.width, th / im.height)
    nw, nh = int(im.width * scale), int(im.height * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    im = im.crop((left, top, left + tw, top + th))
    out = FRAMES_OUT / "rest.webp"
    im.save(out, "WEBP", quality=78, method=6)

    # Face box: use median of near-center clip frames
    near = sorted(rows, key=lambda r: r["radius"])[:20]
    rest = {
        "src": "rest",
        "stem": "portrait",
        "i_src": 0,
        "t": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "radius": 0.0,
        "angle": 0.0,
        "roll": 0.0,
        "ear": 1.0,
        "file": "frames/rest.webp",
        "face_cx": float(np.median([r["face_cx"] for r in near])),
        "face_cy": float(np.median([r["face_cy"] for r in near])),
        "face_rx": float(np.median([r["face_rx"] for r in near])),
        "face_ry": float(np.median([r["face_ry"] for r in near])),
        "w": 848,
        "h": 480,
    }
    print(f"Rest frame {out} ({out.stat().st_size/1024:.0f} KB) face=({rest['face_cx']:.3f},{rest['face_cy']:.3f})")
    return rest


def write_json(rows, rest, origin):
    # Face location for JS (angle-from-head)
    near = sorted(rows, key=lambda r: r["radius"])[:24]
    face = {
        "cx": float(np.median([r["face_cx"] for r in near])),
        "cy": float(np.median([r["face_cy"] for r in near])),
        "rx": float(np.median([r["face_rx"] for r in near])),
        "ry": float(np.median([r["face_ry"] for r in near])),
    }
    yaws = [r["yaw"] for r in rows]
    pitches = [r["pitch"] for r in rows]
    # Per-source radius stats (for inner/outer blend reports)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r["radius"])
    src_stats = {
        s: {
            "n": len(v),
            "r_med": float(np.median(v)),
            "r_min": float(np.min(v)),
            "r_max": float(np.max(v)),
        }
        for s, v in by_src.items()
    }
    frames = []
    all_rows = list(rows)
    if rest:
        all_rows = [rest] + all_rows
    for i, r in enumerate(all_rows):
        frames.append(
            {
                "i": i,
                "yaw": round(r["yaw"], 4),
                "pitch": round(r["pitch"], 4),
                "radius": round(r.get("radius", math.hypot(r["yaw"], r["pitch"])), 4),
                "angle": round(r.get("angle", math.degrees(math.atan2(r["pitch"], r["yaw"]))), 3),
                "file": r["file"],
                "src": r["src"],
            }
        )
    max_abs_yaw = float(max(abs(x) for x in yaws))
    max_abs_pitch = float(max(abs(x) for x in pitches))
    r_max = float(max(r["radius"] for r in rows))
    data = {
        "width": 848,
        "height": 480,
        "fps": 24,
        "originYaw": origin[0],
        "originPitch": origin[1],
        "maxAbsYaw": max_abs_yaw,
        "maxAbsPitch": max_abs_pitch,
        "rMax": r_max,
        "face": face,
        "sources": src_stats,
        "restIndex": 0 if rest else int(np.argmin([r["radius"] for r in rows])),
        "frames": frames,
    }
    path = ROOT / "frames.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {path}  ({path.stat().st_size/1024:.1f} KB, {len(frames)} frames)")
    print(f"  maxAbsYaw={max_abs_yaw:.2f} maxAbsPitch={max_abs_pitch:.2f} rMax={r_max:.2f}")
    print(f"  face cx,cy,rx,ry={face}")
    print(f"  sources: {src_stats}")
    return data


def coverage_plot(rows, data, out_path: Path):
    colors = {
        "outer": "#c4a574",
        "inner": "#6ec4c8",
        "rest": "#f3ece0",
    }
    # leftover names get a palette
    extra_palette = ["#e07a5f", "#81b29a", "#9b8ec4", "#f2cc8f"]
    ei = 0
    for s in {r["src"] for r in rows}:
        if s not in colors:
            colors[s] = extra_palette[ei % len(extra_palette)]
            ei += 1

    rmax = data["rMax"]
    yaw_lim = max(data["maxAbsYaw"], rmax) * 1.18
    pitch_lim = max(data["maxAbsPitch"], rmax) * 1.18
    lim = max(yaw_lim, pitch_lim)

    # Grid coverage
    n = 161
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.array([[r["yaw"], r["pitch"]] for r in rows])
    # distance to nearest frame
    # chunked to stay light
    dist = np.full(XX.shape, np.inf)
    for i in range(0, XX.size, 800):
        sl = slice(i, i + 800)
        q = np.column_stack([XX.ravel()[sl], YY.ravel()[sl]])
        # (q, 2) vs (P, 2)
        d = np.sqrt(((q[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        dist.ravel()[sl] = d

    # Threshold: 3 degrees or 12% of rmax, whichever is larger-ish
    thr = max(2.5, 0.12 * rmax)
    # Regions inside the observed circle (plus a little)
    RR = np.sqrt(XX**2 + YY**2)
    ANG = np.degrees(np.arctan2(YY, XX))  # 0=right, 90=up
    inside = RR <= rmax * 1.02

    def frac(mask):
        m = mask & inside
        if m.sum() == 0:
            return 0.0, 0
        covered = (dist <= thr) & m
        return float(covered.sum() / m.sum()), int(m.sum())

    interior = RR <= rmax * 0.45
    mid = (RR > rmax * 0.45) & (RR <= rmax * 0.75)
    outer_r = RR > rmax * 0.75
    # diagonal wedges ±22° around 45, 135, -45, -135
    def wedge(center, half=22):
        a = np.abs((ANG - center + 180) % 360 - 180)
        return a <= half

    diag = wedge(45) | wedge(135) | wedge(-45) | wedge(-135)
    cardinal = wedge(0) | wedge(90) | wedge(180) | wedge(-90)

    metrics = {}
    for name, mask in [
        ("all", np.ones_like(RR, dtype=bool)),
        ("interior r<0.45 rmax", interior),
        ("mid 0.45–0.75", mid),
        ("outer r>0.75", outer_r),
        ("diagonals (±22° of 45s)", diag),
        ("cardinals (±22° of axes)", cardinal),
    ]:
        f, ncell = frac(mask)
        metrics[name] = {"frac": f, "cells": ncell}
        print(f"  coverage {name}: {f*100:.1f}%  (thr={thr:.2f}°, cells={ncell})")

    # Gap description: uncovered cells clustered
    uncovered = inside & (dist > thr)
    gap_notes = []
    if uncovered.any():
        # mean angle/radius of uncovered
        for name, mask in [
            ("interior", interior),
            ("mid ring", mid),
            ("outer ring", outer_r),
            ("NE diagonal", wedge(45)),
            ("NW diagonal", wedge(135)),
            ("SW diagonal", wedge(-135)),
            ("SE diagonal", wedge(-45)),
        ]:
            m = uncovered & mask
            if m.sum() > 40:
                gap_notes.append(
                    f"{name}: {100*m.sum()/max(1,(mask&inside).sum()):.0f}% of region uncovered"
                )

    fig = plt.figure(figsize=(11, 10), facecolor="#110f0d")
    ax = fig.add_axes([0.1, 0.08, 0.72, 0.82])
    ax.set_facecolor("#1a1714")
    # heatmap of nearest-neighbour distance
    cmap = plt.cm.magma
    cm = cmap(np.linspace(0, 1, 256))
    # darken
    from matplotlib.colors import ListedColormap

    heat = ax.pcolormesh(
        XX, YY, np.clip(dist, 0, thr * 3),
        cmap="magma", shading="auto", alpha=0.85, zorder=1,
    )
    # uncovered contour
    ax.contour(XX, YY, dist, levels=[thr], colors=["#f3ece0"], linewidths=0.8, alpha=0.7, zorder=2)

    for rad, ls in [(0.45 * rmax, ":"), (0.75 * rmax, ":"), (rmax, "-")]:
        ax.add_patch(
            plt.Circle((0, 0), rad, fill=False, ec="#f3ece0", lw=0.7, ls=ls, alpha=0.4, zorder=3)
        )
    ax.axhline(0, color="#f3ece0", lw=0.4, alpha=0.3, zorder=3)
    ax.axvline(0, color="#f3ece0", lw=0.4, alpha=0.3, zorder=3)
    for a in (45, 135, -45, -135):
        rad = math.radians(a)
        ax.plot(
            [0, lim * math.cos(rad)],
            [0, lim * math.sin(rad)],
            color="#f3ece0", lw=0.4, alpha=0.2, zorder=3,
        )

    # points
    handles = []
    for src in sorted({r["src"] for r in rows}, key=lambda s: (s != "outer", s != "inner", s)):
        rs = [r for r in rows if r["src"] == src]
        ax.scatter(
            [r["yaw"] for r in rs],
            [r["pitch"] for r in rs],
            c=colors[src], s=18, alpha=0.9, linewidths=0, zorder=4, label=src,
        )
        handles.append(
            mpatches.Patch(color=colors[src], label=f"{src}  (n={len(rs)})")
        )
    ax.scatter([0], [0], c="#f3ece0", s=48, marker="+", zorder=5, linewidths=1.4)

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("yaw  (°,  + = viewer's right)", color="#c8bfb0")
    ax.set_ylabel("pitch  (°,  + = up)", color="#c8bfb0")
    ax.tick_params(colors="#8a8276")
    for sp in ax.spines.values():
        sp.set_color("#8a8276")
    ax.set_title(
        "Maison Lunaire  ·  gaze pose coverage",
        color="#f3ece0",
        fontname="DejaVu Serif",
        fontsize=16,
        pad=12,
    )
    ax.legend(handles=handles, loc="upper right", facecolor="#110f0d", edgecolor="#8a8276",
              labelcolor="#f3ece0", fontsize=9)

    cax = fig.add_axes([0.84, 0.08, 0.02, 0.4])
    cb = fig.colorbar(heat, cax=cax)
    cb.set_label("nearest-frame distance (°)", color="#c8bfb0")
    cb.ax.yaxis.set_tick_params(color="#8a8276")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8a8276")

    # metrics panel
    txt = [f"covered if nearest ≤ {thr:.1f}°", ""]
    for name, m in metrics.items():
        txt.append(f"{name}:  {m['frac']*100:.1f}%")
    if gap_notes:
        txt.append("")
        txt.append("Gaps:")
        txt.extend("· " + g for g in gap_notes)
    fig.text(
        0.84, 0.92, "\n".join(txt),
        va="top", ha="left", color="#c8bfb0", fontsize=8, family="DejaVu Sans",
        transform=fig.transFigure,
    )
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Wrote {out_path}")
    return metrics, gap_notes, thr


def write_report(metrics, gap_notes, thr, n_in, n_out, n_kept, n_dropped, payload, data):
    p = ROOT / "tools" / "coverage_report.txt"
    lines = [
        f"threshold: {thr:.2f}°",
        f"frames kept: {n_kept} (outer+inner+other)  dropped: {n_dropped}",
        f"payload webp: {payload/1e6:.2f} MB",
        f"rMax: {data['rMax']:.2f}  maxYaw: {data['maxAbsYaw']:.2f}  maxPitch: {data['maxAbsPitch']:.2f}",
        "",
        "Coverage (fraction of yaw-pitch cells with a frame within threshold):",
    ]
    for name, m in metrics.items():
        lines.append(f"  {name}: {m['frac']*100:.1f}%")
    lines.append("")
    if gap_notes:
        lines.append("Gaps:")
        lines.extend("  " + g for g in gap_notes)
    else:
        lines.append("No significant clustered gaps.")
    p.write_text("\n".join(lines) + "\n")
    print(p.read_text())


def main():
    WORK.mkdir(exist_ok=True)
    FRAMES_OUT.mkdir(exist_ok=True)
    import mediapipe as mp

    clips = discover_clips()
    print("Clips:", clips)
    landmarker = make_landmarker()
    all_rows = []
    for clip in clips:
        dest = Path(f"/tmp/gaze-frames-{clip['src']}")
        # keep compatibility with earlier outer extract path
        if clip["src"] == "outer" and Path("/tmp/gaze-frames-raw").exists() and list(Path("/tmp/gaze-frames-raw").glob("f_*.png")):
            dest = Path("/tmp/gaze-frames-raw")
            print(f"reuse outer pngs at {dest}")
        else:
            extract(clip, dest)
        rows = measure_clip(landmarker, mp, clip, dest)
        all_rows.extend(rows)

    use, ys, ps = choose_convention(all_rows)
    apply_convention(all_rows, use, ys, ps)
    origin = origin_and_recenter(all_rows)

    # Re-print sanity with recentered values
    outer = [r for r in all_rows if r["src"] == "outer"]
    print("\nRecentered outer clip at 0/1/2/3/4/8s:")
    for t in (0, 1, 2, 3, 4, 8):
        cand = [r for r in outer if abs(r["t"] - t) <= 0.15]
        if not cand:
            continue
        r = cand[len(cand) // 2]
        print(f"  t={r['t']:.2f}  yaw={r['yaw']:+6.2f}  pitch={r['pitch']:+6.2f}  r={r['radius']:.2f}  ang={r['angle']:+6.1f}  ear={r['ear']:.3f}")

    inner = [r for r in all_rows if r["src"] == "inner"]
    if inner:
        print("\nInner clip radius: "
              f"min={min(r['radius'] for r in inner):.2f}  "
              f"med={np.median([r['radius'] for r in inner]):.2f}  "
              f"max={max(r['radius'] for r in inner):.2f}")
        print("Inner at 0/1/2/3/4/8s:")
        for t in (0, 1, 2, 3, 4, 8):
            cand = [r for r in inner if abs(r["t"] - t) <= 0.15]
            if not cand:
                continue
            r = cand[len(cand) // 2]
            print(f"  t={r['t']:.2f}  yaw={r['yaw']:+6.2f}  pitch={r['pitch']:+6.2f}  r={r['radius']:.2f}  ang={r['angle']:+6.1f}  ear={r['ear']:.3f}")

    kept, dropped = filter_rows(all_rows)
    kept = light_dedup(kept, deg=0.50)

    # Encode; if payload too big, lower quality
    payload = encode_webp(kept, quality=64)
    if payload > 6.6e6:
        print("Payload high, re-encoding at q=54")
        payload = encode_webp(kept, quality=54)
    if payload > 6.6e6:
        print("Still high, re-encoding at q=48 and 760px")
        # shrink
        for r in kept:
            im = Image.open(r["png"]).convert("RGB").resize((760, 430), Image.Resampling.LANCZOS)
            tmp = Path(r["png"]).with_suffix(".tmp.png")
            im.save(tmp)
            r["png"] = str(tmp)
            r["w"], r["h"] = 760, 430
        payload = encode_webp(kept, quality=52)

    rest = encode_rest_portrait(kept)
    payload = sum(p.stat().st_size for p in FRAMES_OUT.glob("*.webp"))
    print(f"Total webp including rest: {payload/1e6:.2f} MB")

    data = write_json(kept, rest, origin)
    metrics, gap_notes, thr = coverage_plot(kept, data, ROOT / "coverage.png")
    write_report(metrics, gap_notes, thr, len(inner), len(outer), len(kept), len(dropped), payload, data)

    # Save a few diagnostic annotated frames
    diag = WORK / "diag"
    diag.mkdir(exist_ok=True)
    for label, t, src in [
        ("outer-right", 1.0, "outer"),
        ("outer-up", 2.0, "outer"),
        ("outer-left", 3.0, "outer"),
        ("outer-down", 4.0, "outer"),
        ("outer-center", 0.0, "outer"),
        ("inner-t1", 1.0, "inner"),
        ("inner-t2", 2.0, "inner"),
    ]:
        cand = [r for r in kept if r["src"] == src and abs(r["t"] - t) <= 0.2]
        if not cand:
            cand = [r for r in all_rows if r["src"] == src and abs(r["t"] - t) <= 0.2]
        if not cand:
            continue
        r = cand[len(cand) // 2]
        im = Image.open(r["png"]).convert("RGB")
        im.save(diag / f"{label}.jpg", quality=85)
        print(f"  diag {label}: yaw={r['yaw']:+.1f} pitch={r['pitch']:+.1f} {r.get('file','')}")


if __name__ == "__main__":
    main()
