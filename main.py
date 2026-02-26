import cv2
import mediapipe as mp
import pyvirtualcam
import os
import time
import psutil
import numpy as np

# --- 配置参数 ---
PROCESS_NAME = "zoom.us"
WAVE_COUNT_TARGET = 6   # 3次来回 = 6次方向切换
TIME_WINDOW = 1.5       # 必须在1.5秒内完成挥手
# 适配 Mac 摄像头
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 初始化 MediaPipe (采用底层导入确保 Mac 兼容性)
from mediapipe.python.solutions import hands as mp_hands
hands_engine = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def is_zoom_running():
    """性能优化：检查 Zoom 是否在运行"""
    for proc in psutil.process_iter(['name']):
        try:
            if PROCESS_NAME in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def run_gesture_controller():
    # 使用 AVFoundation 显式打开 Mac 摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    # 强制设置分辨率，避免 ValueError
    cap.set(3, TARGET_WIDTH)  # 3 对应 CAP_FRAME_WIDTH
    cap.set(4, TARGET_HEIGHT) # 4 对应 CAP_FRAME_HEIGHT
    
    # 获取实际生效的分辨率
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    if width == 0 or height == 0:
        print("错误：无法初始化摄像头，请检查系统隐私设置中的权限。")
        return

    # 动态识别变量
    x_history = []
    directions = [] 
    
    print(f"🚀 AI 引擎已启动。分辨率: {width}x{height}")
    print(f"💡 请在 Zoom 摄像头设置中选择 'Python Virtual Camera'。")

    # 启动虚拟相机隧道
    with pyvirtualcam.Camera(width, height, 30, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        while True:
            # 性能优化：Zoom 不在运行时进入休眠
            if not is_zoom_running():
                time.sleep(2)
                continue

            success, frame = cap.read()
            if not success:
                continue

            # 镜像处理并转换颜色空间
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_engine.process(rgb_frame)

            if results.multi_hand_landmarks:
                # 获取手掌中心点 (Landmark 9) 的 X 坐标
                hand = results.multi_hand_landmarks[0]
                current_x = hand.landmark[9].x
                current_time = time.time()
                x_history.append((current_x, current_time))

                # 只保留时间窗口内的记录
                x_history = [h for h in x_history if current_time - h[1] < TIME_WINDOW]

                if len(x_history) > 5:
                    # 计算水平位移方向
                    dx = x_history[-1][0] - x_history[-2][0]
                    if abs(dx) > 0.005: # 过滤细微抖动
                        current_dir = 1 if dx > 0 else -1
                        
                        # 只有方向切换时才计数
                        if not directions or current_dir != directions[-1]:
                            directions.append(current_dir)
                            
                            # 达到挥手次数目标
                            if len(directions) >= WAVE_COUNT_TARGET:
                                print(" 检测到‘拜拜’手势！正在执行强制关闭...")
                                os.system(f"pkill -9 {PROCESS_NAME}")
                                directions = [] # 重置状态
                                x_history = []
            else:
                # 没看到手时，若超时则重置计数
                if x_history and time.time() - x_history[-1][1] > 0.5:
                    directions = []
                    x_history = []

            # 核心步骤：将处理后的帧发送到虚拟摄像头（无 imshow，实现隐藏）
            try:
                vcam.send(frame)
                vcam.sleep_until_next_frame()
            except Exception as e:
                print(f"虚拟相机发送异常: {e}")
                break

    cap.release()

if __name__ == "__main__":
    try:
        run_gesture_controller()
    except KeyboardInterrupt:
        print("\nAI 助手已手动停止。")