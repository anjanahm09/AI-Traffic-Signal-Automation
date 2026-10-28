import pygame
import cv2
import numpy as np
import time

# ==============================
# Pygame Setup
# ==============================
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Traffic Signal Control System")

# Colors
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

font = pygame.font.SysFont(None, 30)

# Signal positions (4 roads)
SIGNAL_POSITIONS = [(400, 100), (700, 400), (400, 700), (100, 400)]

# Signal states
RED_STATE = 0
GREEN_STATE = 1
YELLOW_STATE = 2

# Timing control
RED_TIME = 10
YELLOW_TIME = 3
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30

# ==============================
# Load YOLOv3
# ==============================
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("classes.names", "r") as f:
    classes = f.read().strip().split("\n")

# Video input
cap = cv2.VideoCapture('1.mp4')

# ==============================
# State Variables
# ==============================
current_states = [RED_STATE] * 4
signal_timers = [time.time()] * 4
vehicle_counts = [0] * 4
emergency_detected = [False] * 4
active_road = 0

remaining_times = [RED_TIME] * 4
green_durations = [MIN_GREEN_TIME] * 4  # assigned when road turns green

# ==============================
# Main Loop
# ==============================
running = True
while running:
    screen.fill(WHITE)
    current_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 416))
    height, width, _ = frame.shape

    # Prepare image for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Reset counts
    vehicle_counts = [0] * 4
    emergency_detected = [False] * 4

    # Process YOLO detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]
                detected_road = np.random.randint(0, 4)  # TODO: Replace with region mapping

                if label in ["car", "truck", "bus", "motorbike"]:
                    vehicle_counts[detected_road] += 1
                elif label in ["ambulance", "fire truck", "police car"]:
                    emergency_detected[detected_road] = True

    # ==============================
    # Traffic Signal Logic
    # ==============================
    for i in range(4):
        elapsed_time = current_time - signal_timers[i]

        if current_states[i] == RED_STATE and i == active_road:
            remaining_times[i] = max(0, RED_TIME - elapsed_time)
            if elapsed_time > RED_TIME:
                # Turn GREEN
                current_states[i] = GREEN_STATE
                signal_timers[i] = current_time

                # Assign green duration ONCE
                if emergency_detected[i]:
                    green_durations[i] = MAX_GREEN_TIME
                else:
                    green_durations[i] = min(
                        MAX_GREEN_TIME,
                        max(MIN_GREEN_TIME, vehicle_counts[i] * 2)
                    )

                remaining_times[i] = green_durations[i]
                print(f"Road {i+1} GREEN for {green_durations[i]}s "
                      f"(Vehicles: {vehicle_counts[i]}, Ambulance: {emergency_detected[i]})")

        elif current_states[i] == GREEN_STATE:
            remaining_times[i] = max(0, green_durations[i] - elapsed_time)
            if elapsed_time > green_durations[i]:
                current_states[i] = YELLOW_STATE
                signal_timers[i] = current_time
                remaining_times[i] = YELLOW_TIME

        elif current_states[i] == YELLOW_STATE:
            remaining_times[i] = max(0, YELLOW_TIME - elapsed_time)
            if elapsed_time > YELLOW_TIME:
                current_states[i] = RED_STATE
                signal_timers[i] = current_time
                remaining_times[i] = RED_TIME
                active_road = (active_road + 1) % 4

    # ==============================
    # Drawing UI
    # ==============================
    for idx, pos in enumerate(SIGNAL_POSITIONS):
        if current_states[idx] == RED_STATE:
            pygame.draw.circle(screen, RED, pos, 30)
        elif current_states[idx] == GREEN_STATE:
            pygame.draw.circle(screen, GREEN, pos, 30)
        elif current_states[idx] == YELLOW_STATE:
            pygame.draw.circle(screen, YELLOW, pos, 30)

        vehicle_text = font.render(f"Road {idx+1}: {vehicle_counts[idx]} vehicles", True, BLACK)
        screen.blit(vehicle_text, (pos[0] - 100, pos[1] + 40))

        time_text = font.render(f"Time: {int(remaining_times[idx])}s", True, BLACK)
        screen.blit(time_text, (pos[0] - 100, pos[1] + 60))

    # Display camera feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    frame_surface = pygame.transform.scale(frame_surface, (300, 300))
    screen.blit(frame_surface, (250, 250))

    # Update display
    pygame.display.flip()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
