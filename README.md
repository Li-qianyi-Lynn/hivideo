HiVideo – Zoom Hand-Wave Force Quit Helper
==========================================

HiVideo is a small utility that uses your webcam and a virtual camera to detect a fast “bye‑bye” hand‑wave gesture and force‑quit Zoom automatically.  
It is designed for macOS and integrates with Zoom by exposing a virtual camera named **Python Virtual Camera**.

---

Features
--------

- **Gesture-based Zoom shutdown**: Wave your hand quickly in front of the camera to trigger a forced Zoom quit.
- **Low-latency hand tracking**: Uses MediaPipe Hands for real-time hand landmark detection.
- **Virtual camera integration**: Streams processed frames to a virtual camera via `pyvirtualcam`, so Zoom can use it as a regular video source.
- **Mac‑specific camera handling**: Uses `AVFoundation` (`cv2.CAP_AVFOUNDATION`) to open the built‑in camera on macOS.
- **Process-aware**: Periodically checks whether the `zoom.us` process is running and sleeps when Zoom is not open to save CPU.

---

How It Works
------------

1. The script opens your Mac’s camera at a target resolution of 1280×720 using OpenCV.
2. Each frame is passed through MediaPipe Hands to track the position of one hand.
3. The X‑coordinate of landmark 9 (center of the palm area) is monitored over a short time window (default: 1.5 seconds).
4. When enough left–right direction changes (default: 6) are detected within the time window, the script interprets this as a rapid “bye‑bye” wave.
5. It then runs `pkill -9 zoom.us` to force‑quit the Zoom process.
6. All frames are sent to a virtual camera created by `pyvirtualcam`, which you can select in Zoom as your video source.

---

Requirements
------------

- **Operating System**: macOS
- **Python**: 3.8+ (recommended)
- **Dependencies**:
  - `opencv-python`
  - `mediapipe`
  - `pyvirtualcam`
  - `psutil`
  - `numpy`

You can install them with:

```bash
pip install opencv-python mediapipe pyvirtualcam psutil numpy
```

---

Installation
------------

1. **Clone or download this repository** into a folder on your Mac.
2. (Optional but recommended) **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install opencv-python mediapipe pyvirtualcam psutil numpy
   ```

4. Make sure Zoom is installed and that you have allowed **Camera access** for your terminal or Python environment in **System Settings → Privacy & Security → Camera**.

---

Usage
-----

1. Open a terminal and navigate to the project directory.

2. Run the script:

   ```bash
   python3 main.py
   ```

3. You should see logs similar to:

   ```text
   AI engine started
   ```

4. Open **Zoom**, go to **Settings → Video**, and select **Python Virtual Camera** as the camera source.

5. When you want to quickly leave a meeting, wave your hand rapidly left and right in front of the camera.  
   
---

Configuration
-------------

You can adjust several parameters at the top of `main.py`:

```python
PROCESS_NAME = "zoom.us"      # Process name to monitor and kill
WAVE_COUNT_TARGET = 6         # Number of direction switches required
TIME_WINDOW = 1.5             # Time window (seconds) to complete the wave
TARGET_WIDTH = 1280           # Target camera width
TARGET_HEIGHT = 720           # Target camera height
```

- **`WAVE_COUNT_TARGET`**: Increase to make the gesture harder to trigger, decrease to make it easier.
- **`TIME_WINDOW`**: Decrease for a faster, more intense wave; increase for a slower wave.
- **`PROCESS_NAME`**: Change this if you want to control a different app (process name must match).

---

Limitations & Notes
-------------------

- This script is currently tuned for **macOS + Zoom** and may require adaptation for other platforms or video apps.
- Because it uses `pkill -9`, the target app is force‑killed without a graceful shutdown.
- The gesture detection is based on one hand and horizontal movement; strong lighting and clear hand visibility will improve reliability.
- If the camera cannot be initialized, check macOS privacy settings for camera permissions.

---

