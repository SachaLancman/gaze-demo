# Maison Lunaire — gaze demo

A concept site for the fictional watch house Maison Lunaire. A man follows your cursor by showing the pre-baked video frame whose head direction is closest to the angle from his face to the pointer. It is not 3D.

Live: https://sachalancman.github.io/gaze-demo/

Companion lookbook: https://sachalancman.github.io/watch-demo/

## How it works

Two AI clips (an outer head-circle and a smaller inner circle) are measured with MediaPipe Face Landmarker. Each kept frame is stored as a WebP plus a `(yaw, pitch)` pose. At runtime the page:

1. Takes the **angle** from his head on screen to the cursor, and the **distance** from the head (near → inner poses, far → outer poses; cursor on the face → rest frame, looking at you).
2. Glides to that pose, always along the **shorter arc** around the loop.
3. Draws the nearest pre-baked frame on a canvas. The video never plays.
4. After 2.5s of no movement — and on phones — he slowly looks around on his own.
5. `prefers-reduced-motion` shows only the rest frame.

To merge another clip later, drop it in `media/` (anything except `gaze-loop.mp4`) and re-run `tools/process_frames.py`.

Model and footage are AI-generated. This is a concept demo, not a store.
