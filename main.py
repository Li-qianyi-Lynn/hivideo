import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pyvirtualcam
import os
import time
import psutil
import numpy as np

# config
PROCESS_NAME = "zoom.us"
WAVE_COUNT_TARGET = 6   # 3 times up and down = 6 times direction change
TIME_WINDOW = 1.5       # 
# mac camera
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# hand model path (MediaPipe Tasks need separate .task model file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# hand 21 landmarks connections (for hand skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),            # palm
]
# palm area contour (wrist + 4 finger roots, for red fill)
PALM_INDEXES = [0, 5, 9, 13, 17]


def to_pixel(x_norm, y_norm, w, h):
    """normalize coordinate [0,1] to pixel coordinate"""
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)


def draw_hand_red(frame, hand_landmarks, width, height):
    """draw hand: red palm + green skeleton on the frame"""
    pts = [to_pixel(lm.x, lm.y, width, height) for lm in hand_landmarks]
    # red palm: fill with polygon
    palm_pts = np.array([pts[i] for i in PALM_INDEXES], dtype=np.int32)
    cv2.fillPoly(frame, [palm_pts], (0, 0, 255))  # BGR red
    # palm outline
    cv2.polylines(frame, [palm_pts], True, (0, 0, 200), 2)
    # palm center point (Landmark 9) draw a red circle
    cv2.circle(frame, pts[9], 12, (0, 0, 255), -1)
    cv2.circle(frame, pts[9], 12, (255, 255, 255), 1)
    # green skeleton lines
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    # joint points draw small circles
    for (px, py) in pts:
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)


def ensure_hand_model():
    """if local hand_landmarker.task is not found, download it"""
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH
    except Exception as e:
        print("download failed:", e)
        print("please download and put it in the project directory:")
        print("  ", MODEL_URL)
        print("  save as:", MODEL_PATH)
        return None


def is_zoom_running():
    """performance optimization: check if Zoom is running"""
    for proc in psutil.process_iter(['name']):
        try:
            if PROCESS_NAME in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def run_gesture_controller():
    model_path = ensure_hand_model()
    if not model_path:
        return

    # use AVFoundation explicitly open Mac camera
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    cap.set(3, TARGET_WIDTH)
    cap.set(4, TARGET_HEIGHT)

    width = int(cap.get(3))
    height = int(cap.get(4))

    if width == 0 or height == 0:
        print("error: cannot initialize camera, please check the permissions in the system privacy settings.")
        return

    x_history = []
    directions = []
    frame_index = 0

    print(f"AI engine started. resolution: {width}x{height}")
    print(f"please select 'Python Virtual Camera' in the Zoom camera settings.")

    with vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
    ) as landmarker:
        with pyvirtualcam.Camera(width, height, 30, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            while True:
                if not is_zoom_running():
                    time.sleep(2)
                    continue

                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # VIDEO mode needs millisecond timestamp (increment)
                timestamp_ms = frame_index * 1000 // 30  # milliseconds, 30fps
                frame_index += 1

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    # get the first hand, Landmark 9 is the palm area center (same as old API)
                    hand = result.hand_landmarks[0]
                    current_x = hand[9].x
                    current_time = time.time()
                    x_history.append((current_x, current_time))
                    x_history = [h for h in x_history if current_time - h[1] < TIME_WINDOW]

                    if len(x_history) > 5:
                        dx = x_history[-1][0] - x_history[-2][0]
                        if abs(dx) > 0.005:
                            current_dir = 1 if dx > 0 else -1
                            if not directions or current_dir != directions[-1]:
                                directions.append(current_dir)
                                if len(directions) >= WAVE_COUNT_TARGET:
                                    print(" detected 'bye-bye' gesture! executing force close...")
                                    os.system(f"pkill -9 {PROCESS_NAME}")
                                    directions = []
                                    x_history = []
                else:
                    if x_history and time.time() - x_history[-1][1] > 0.5:
                        directions = []
                        x_history = []

                display_frame = frame.copy()
                # detected hand: red palm + green skeleton
                if result.hand_landmarks:
                    draw_hand_red(display_frame, result.hand_landmarks[0], width, height)
                cv2.putText(
                    display_frame,
                    f"Wave count: {len(directions)}/{WAVE_COUNT_TARGET}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("HiVideo - Camera", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("user pressed q to exit.")
                    break

                try:
                    vcam.send(frame)
                    vcam.sleep_until_next_frame()
                except Exception as e:
                    print(f"virtual camera send exception: {e}")
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_gesture_controller()
    except KeyboardInterrupt:
        print("\nAI assistant manually stopped.")
