

\# AI-Based Traffic Signal Automation System



\## Project Overview

An academic project exploring AI-based vehicle detection and traffic signal automation. The goal is to use detected traffic conditions to support adaptive signal timing and give priority to emergency vehicles.



\## Technologies

\- Python

\- OpenCV

\- YOLO-based object detection

\- Pygame (where used by the implementation)



\## Features

\- Vehicle detection from traffic video

\- Object detection using YOLO model configurations

\- Prototype traffic signal visualization

\- Experimental emergency-vehicle detection in selected scripts



\## Project Structure

\- `maincode.py` and related scripts: vehicle-detection prototypes

\- `models/` or configuration files: model configuration and class labels

\- `templates/` and `static/`: web interface assets, if included



\## Setup

1\. Install Python 3.

2\. Install the required Python packages for the script you intend to run.

3\. Obtain compatible model weights separately and place them in the paths expected by the script.

4\. Provide a traffic video file and update the script's input path.

5\. Run the selected Python script.



\## Limitations

This is an academic prototype. The scripts may have different dependencies and input paths. Accurate lane-level vehicle counts and adaptive signal behavior require validation and testing. Emergency-vehicle distance estimates require camera calibration.



\## Note

Model weights, large videos, datasets, and private databases are not included in this repository.



