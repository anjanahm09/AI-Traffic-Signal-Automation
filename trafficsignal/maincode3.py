import pygame
import cv2
import numpy as np
import time
from ultralytics import YOLO
import math

# -------------------------
# Configuration
# -------------------------
VIDEO_SOURCE = "12.mp4"       # or 0 for webcam
YOLOV3_CFG = "yolov3.cfg"
YOLOV3_WEIGHTS = "yolov3.weights"
YOLOV3_NAMES = "coco.names"

YOLOV8_MODEL_PATH = "best.pt"  # your trained model
FRAME_SIZE = 416                           # YOLOv3 expects 416x416
CONF_THRESHOLD = 0.5

NUM_ROADS = 4
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30
RED_TIME = 10
YELLOW_TIME = 3

KNOWN_HEIGHT = 2.5   # meters (approx ambulance height)
FOCAL_LENGTH = 800   # pixels (calibrate)


def estimate_distance(bbox_height_pixels):
    """Pinhole camera model for ambulance distance estimation."""
    if bbox_height_pixels > 0:
        return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height_pixels
    return -1


# -------------------------
# Load YOLOv3 (vehicle detection)
# -------------------------
net = cv2.dnn.readNet(YOLOV3_WEIGHTS, YOLOV3_CFG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(YOLOV3_NAMES, "r") as f:
    classes = f.read().strip().split("\n")

vehicle_classes = {"car", "bus", "truck", "motorbike", "motorcycle"}

# -------------------------
# Load YOLOv8 (ambulance detection)
# -------------------------
yolo_amb = YOLO(YOLOV8_MODEL_PATH)

# -------------------------
# Pygame setup
# -------------------------
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hybrid YOLOv3+YOLOv8 Traffic System")
font = pygame.font.SysFont(None, 26)

# -------------------------
# Traffic signals
# -------------------------
RED_STATE = 0
GREEN_STATE = 1
YELLOW_STATE = 2
current_states = [RED_STATE] * NUM_ROADS
signal_timers = [time.time()] * NUM_ROADS
remaining_times = [RED_TIME] * NUM_ROADS
green_durations = [MIN_GREEN_TIME] * NUM_ROADS
active_road = 0

vehicle_counts = [0] * NUM_ROADS
emergency_detected = [False] * NUM_ROADS
ambulance_distances = [-1] * NUM_ROADS

SIGNAL_POSITIONS = [(400, 100), (700, 400), (400, 700), (100, 400)]


# -------------------------
# Helper: assign detection to road
# -------------------------
def map_to_road(cx, cy, w, h):
    if cy < h / 3:
        return 0
    elif cx > 2 * w / 3:
        return 1
    elif cy > 2 * h / 3:
        return 2
    else:
        return 3


# -------------------------
# Main Loop
# -------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
running = True
while running:
    screen.fill((255, 255, 255))
    now = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    vehicle_counts = [0] * NUM_ROADS
    emergency_detected = [False] * NUM_ROADS
    ambulance_distances = [-1] * NUM_ROADS

    # -------------------------
    # YOLOv3 vehicle detection
    # -------------------------
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (FRAME_SIZE, FRAME_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                label = classes[class_id]
                if label in vehicle_classes:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    road = map_to_road(center_x, center_y, w, h)
                    vehicle_counts[road] += 1

    # -------------------------
    # YOLOv8 ambulance detection
    # -------------------------
    results = yolo_amb(frame)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            bbox_h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = yolo_amb.names[int(cls_id)].lower()
            if class_name == "ambulance":
                road = map_to_road(cx, cy, w, h)
                emergency_detected[road] = True
                ambulance_distances[road] = estimate_distance(bbox_h)

    # -------------------------
    # Signal logic
    # -------------------------
    for i in range(NUM_ROADS):
        elapsed = now - signal_timers[i]

        if current_states[i] == RED_STATE and i == active_road:
            remaining_times[i] = max(0, RED_TIME - elapsed)
            if elapsed > RED_TIME:
                current_states[i] = GREEN_STATE
                signal_timers[i] = now
                if emergency_detected[i]:
                    green_durations[i] = MAX_GREEN_TIME
                else:
                    green_durations[i] = min(MAX_GREEN_TIME, max(MIN_GREEN_TIME, vehicle_counts[i] * 2))
                remaining_times[i] = green_durations[i]
                print(f"Road {i+1} -> GREEN {green_durations[i]}s, vehicles={vehicle_counts[i]}")

        elif current_states[i] == GREEN_STATE:
            remaining_times[i] = max(0, green_durations[i] - elapsed)
            if elapsed > green_durations[i]:
                current_states[i] = YELLOW_STATE
                signal_timers[i] = now
                remaining_times[i] = YELLOW_TIME

        elif current_states[i] == YELLOW_STATE:
            remaining_times[i] = max(0, YELLOW_TIME - elapsed)
            if elapsed > YELLOW_TIME:
                current_states[i] = RED_STATE
                signal_timers[i] = now
                remaining_times[i] = RED_TIME
                active_road = (active_road + 1) % NUM_ROADS

    # Emergency override
    if any(emergency_detected):
        amb_road = emergency_detected.index(True)
        if active_road != amb_road:
            active_road = amb_road
            now = time.time()
            for j in range(NUM_ROADS):
                current_states[j] = RED_STATE
                signal_timers[j] = now
                remaining_times[j] = RED_TIME
            current_states[active_road] = GREEN_STATE
            signal_timers[active_road] = now
            green_durations[active_road] = MAX_GREEN_TIME
            remaining_times[active_road] = green_durations[active_road]
            print(f"🚑 Ambulance detected Road {active_road+1} — GREEN {MAX_GREEN_TIME}s")

    # -------------------------
    # Draw Pygame UI
    # -------------------------
    for idx, pos in enumerate(SIGNAL_POSITIONS):
        color = (255, 0, 0)
        if current_states[idx] == GREEN_STATE:
            color = (0, 255, 0)
        elif current_states[idx] == YELLOW_STATE:
            color = (255, 255, 0)
        pygame.draw.circle(screen, color, pos, 30)

        vc_text = font.render(f"Road {idx+1}: {vehicle_counts[idx]} vehicles", True, (0, 0, 0))
        screen.blit(vc_text, (pos[0] - 110, pos[1] + 40))

        rt_text = font.render(f"Time: {int(remaining_times[idx])}s", True, (0, 0, 0))
        screen.blit(rt_text, (pos[0] - 110, pos[1] + 60))

        if ambulance_distances[idx] > 0:
            amb_text = font.render(f"Ambulance: {ambulance_distances[idx]:.1f}m", True, (0, 0, 255))
            screen.blit(amb_text, (pos[0] - 110, pos[1] + 80))

    # Show live camera feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (300, 300))
    screen.blit(surf, (250, 250))

    pygame.display.flip()

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
