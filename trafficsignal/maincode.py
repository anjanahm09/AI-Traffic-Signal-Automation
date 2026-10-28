import pygame
import cv2
import numpy as np
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Traffic Signal Control System")

# Define Colors
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)  # For emergency vehicles
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Font for Display
font = pygame.font.SysFont(None, 30)

# Traffic Signal Positions (4 Roads)
SIGNAL_POSITIONS = [(400, 100), (700, 400), (400, 700), (100, 400)]

# Signal States
RED_STATE = 0
GREEN_STATE = 1
YELLOW_STATE = 2

# Timing Control
RED_TIME = 10
YELLOW_TIME = 3
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30
DECREASE_RATE = 0.05

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define video capture (use 0 for webcam)
cap = cv2.VideoCapture('1.mp4')

# Initialize states and timers for each road
current_states = [RED_STATE] * 4
signal_timers = [time.time()] * 4
vehicle_counts = [0] * 4
emergency_detected = [False] * 4
active_road = 0  # Start with road 0

remaining_times = [RED_TIME] * 4

dynamic_green_times = [MIN_GREEN_TIME] * 4


# Game loop
running = True
while running:
    screen.fill(WHITE)
    current_time = time.time()

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 416))
    height, width, _ = frame.shape

    # Prepare image for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Reset vehicle counts and emergency detection
    vehicle_counts = [0] * 4
    emergency_detected = [False] * 4

    # Process YOLOv3 detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]
                detected_road = np.random.randint(0, 4)

                if label in ["car", "truck", "bus", "motorbike"]:
                    vehicle_counts[detected_road] += 1
                elif label in ["ambulance", "fire truck", "police car"]:
                    emergency_detected[detected_road] = True

    # Handle Emergency Vehicles
    if any(emergency_detected):
        active_road = emergency_detected.index(True)
        current_states = [RED_STATE] * 4
        current_states[active_road] = GREEN_STATE
        dynamic_green_times[active_road] = MAX_GREEN_TIME

    # Update Traffic Signals
    for i in range(4):
        elapsed_time = current_time - signal_timers[i]

        if current_states[i] == RED_STATE and i == active_road:
            remaining_times[i] = max(0, RED_TIME - elapsed_time)
            if elapsed_time > RED_TIME:
                current_states[i] = GREEN_STATE
                signal_timers[i] = current_time
                remaining_times[i] = dynamic_green_times[i]

        elif current_states[i] == GREEN_STATE:
            remaining_times[i] = max(0, dynamic_green_times[i] - elapsed_time)
            if elapsed_time > dynamic_green_times[i]:
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

    # Draw signals, timings, and vehicle counts
    for idx, pos in enumerate(SIGNAL_POSITIONS):
        if current_states[idx] == RED_STATE:
            pygame.draw.circle(screen, RED, pos, 30)
        elif current_states[idx] == GREEN_STATE:
            pygame.draw.circle(screen, GREEN, pos, 30)
        elif current_states[idx] == YELLOW_STATE:
            pygame.draw.circle(screen, YELLOW, pos, 30)

        # Display vehicle count
        vehicle_count_text = font.render(f"Road {idx + 1}: {vehicle_counts[idx]} vehicles", True, BLACK)
        screen.blit(vehicle_count_text, (pos[0] - 100, pos[1] + 40))

        # Display remaining time
        remaining_time_text = font.render(f"Time: {int(remaining_times[idx])}s", True, BLACK)
        screen.blit(remaining_time_text, (pos[0] - 100, pos[1] + 60))

    # Display camera feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    frame_surface = pygame.transform.scale(frame_surface, (300, 300))
    screen.blit(frame_surface, (250, 250))

    # Update Display
    pygame.display.flip()

    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
