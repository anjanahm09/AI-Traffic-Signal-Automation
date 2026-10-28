import pygame
import cv2
import numpy as np
import time
from ultralytics import YOLO  # YOLOv8 from ultralytics

# ========== Pygame Setup ==========
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Signal Control System with YOLOv8 Ambulance Detection")
font = pygame.font.SysFont(None, 30)

# Colors
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Signal positions for 4 roads (you can adjust based on layout)
SIGNAL_POSITIONS = [(400, 100), (700, 400), (400, 700), (100, 400)]

# Signal states
RED_STATE = 0
GREEN_STATE = 1
YELLOW_STATE = 2

# Timing control constants
RED_TIME = 10
YELLOW_TIME = 3
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30

# Distance estimation params
KNOWN_HEIGHT = 2.5  # meters, approx ambulance height
FOCAL_LENGTH = 800  # calibrate this

def estimate_distance(bbox_height):
    if bbox_height > 0:
        return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height
    else:
        return -1

# ========== Load YOLOv8 Model ==========
# Replace 'path/to/your/model.pt' with your custom model that has ambulance
yolo_model = YOLO("best.pt")

# ========== Video Capture ==========
cap = cv2.VideoCapture("1.mp4")  # or use 0 for webcam

# ========== State Variables ==========
current_states = [RED_STATE] * 4
signal_timers = [time.time()] * 4

vehicle_counts = [0] * 4
emergency_detected = [False] * 4
ambulance_distances = [-1] * 4

active_road = 0
remaining_times = [RED_TIME] * 4
green_durations = [MIN_GREEN_TIME] * 4  # per-road green time assigned once on turning green

# ========== Main Loop ==========
running = True
while running:
    screen.fill(WHITE)
    current_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Optionally, resize frame for speed + consistency
    frame_resized = cv2.resize(frame, (640, 640))
    h_frame, w_frame, _ = frame_resized.shape

    # YOLO inference
    results = yolo_model(frame_resized)  # this returns results; adjust depending on your YOLOv8 version

    # Reset counts
    vehicle_counts = [0] * 4
    emergency_detected = [False] * 4
    ambulance_distances = [-1] * 4

    # Process detections
    for res in results:  # depending on API; results[0], etc.
        boxes = res.boxes  # bounding boxes
        cls_ids = res.boxes.cls  # class ids
        # confidence scores if needed: res.boxes.conf

        for box, cls in zip(boxes, cls_ids):
            class_name = yolo_model.names[int(cls)]  # get label name
            # get bounding box dims
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # adjust if needed
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            # compute center or decide road based on region
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # determine which road this belongs to using regions of interest (ROIs)
            # Example simple mapping:
            if cy < h_frame / 3:
                detected_road = 0  # North
            elif cx > 2 * w_frame / 3:
                detected_road = 1  # East
            elif cy > 2 * h_frame / 3:
                detected_road = 2  # South
            else:
                detected_road = 3  # West

            # Count vehicles
            if class_name in ["car", "truck", "bus", "motorbike"]:
                vehicle_counts[detected_road] += 1

            # Ambulance detection
            if class_name == "ambulance":
                emergency_detected[detected_road] = True
                distance = estimate_distance(bbox_h)
                ambulance_distances[detected_road] = distance
                # You may break or continue; maybe multiple ambulances but handle first

    # ========== Compute Green Duration Once ==========
    # When active_road just turned green, green_durations[active_road] should already be assigned
    # We detect transitions from RED→GREEN to trigger assignment

    # For simplicity, we use logic in the update of signal states below

    # ========== Update Signal States ==========
    for i in range(4):
        elapsed = current_time - signal_timers[i]

        if current_states[i] == RED_STATE and i == active_road:
            # waiting during red before green
            remaining_times[i] = max(0, RED_TIME - elapsed)
            if elapsed > RED_TIME:
                # Transition to GREEN
                current_states[i] = GREEN_STATE
                signal_timers[i] = current_time

                # Assign green duration ONCE
                if emergency_detected[i]:
                    green_durations[i] = MAX_GREEN_TIME
                else:
                    # scale green by count
                    green_durations[i] = min(
                        MAX_GREEN_TIME,
                        max(MIN_GREEN_TIME, vehicle_counts[i] * 2)
                    )
                remaining_times[i] = green_durations[i]
                print(f"Road {i+1} turned GREEN for {green_durations[i]}s "
                      f"(Vehicles: {vehicle_counts[i]}, Ambulance: {emergency_detected[i]})")

        elif current_states[i] == GREEN_STATE:
            remaining_times[i] = max(0, green_durations[i] - elapsed)
            if elapsed > green_durations[i]:
                # go to YELLOW
                current_states[i] = YELLOW_STATE
                signal_timers[i] = current_time
                remaining_times[i] = YELLOW_TIME

        elif current_states[i] == YELLOW_STATE:
            remaining_times[i] = max(0, YELLOW_TIME - elapsed)
            if elapsed > YELLOW_TIME:
                # go to RED and move next active road
                current_states[i] = RED_STATE
                signal_timers[i] = current_time
                remaining_times[i] = RED_TIME
                active_road = (active_road + 1) % 4

    # ========== Ambulance Preemption ==========
    # If any ambulance detected, override: set that road green immediately
    if any(emergency_detected):
        amb_road = emergency_detected.index(True)
        if active_road != amb_road:
            # force switching
            active_road = amb_road
            # reset state timers
            for j in range(4):
                current_states[j] = RED_STATE
                signal_timers[j] = current_time
            current_states[active_road] = GREEN_STATE
            # assign green duration
            green_durations[active_road] = MAX_GREEN_TIME
            remaining_times[active_road] = green_durations[active_road]
            print(f"🚑 Ambulance detected on Road {active_road+1}, forcing GREEN for {MAX_GREEN_TIME}s")

    # ========== Draw UI ==========
    for idx, pos in enumerate(SIGNAL_POSITIONS):
        # draw the current signal
        if current_states[idx] == RED_STATE:
            pygame.draw.circle(screen, RED, pos, 30)
        elif current_states[idx] == GREEN_STATE:
            pygame.draw.circle(screen, GREEN, pos, 30)
        elif current_states[idx] == YELLOW_STATE:
            pygame.draw.circle(screen, YELLOW, pos, 30)

        # vehicle count text
        vc_text = font.render(f"Road {idx+1}: {vehicle_counts[idx]} vehicles", True, BLACK)
        screen.blit(vc_text, (pos[0] - 100, pos[1] + 40))

        # remaining time text
        rt = int(remaining_times[idx])
        remaining_text = font.render(f"Time: {rt}s", True, BLACK)
        screen.blit(remaining_text, (pos[0] - 100, pos[1] + 60))

        # ambulance distance if any
        if ambulance_distances[idx] > 0:
            dist_text = font.render(f"Ambulance: {ambulance_distances[idx]:.1f} m", True, BLUE)
            screen.blit(dist_text, (pos[0] - 100, pos[1] + 80))

    # show camera feed small
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # for pygame display, convert
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
    frame_surface = pygame.transform.scale(frame_surface, (300, 300))
    screen.blit(frame_surface, (250, 250))

    pygame.display.flip()

    # event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# cleanup
cap.release()
pygame.quit()
