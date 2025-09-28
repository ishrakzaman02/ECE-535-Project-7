# ECE-535-Project-7
# Smart Doorbell using Raspberry Pi and ML
# Ishrak Zaman - 33812024
# Armaan Khan - 33574580
# Anika Badkul - 33746454

## Motivation
Smart home devices are increasingly important for convenience and security. A smart doorbell that can recognize household members or unknown visitors provides real-time alerts without relying heavily on cloud infrastructure, making it privacy-friendly and efficient.

## Design Goals
- Deploy lightweight ML models directly on Raspberry Pi.
- Achieve real-time person/face detection and classification.
- Keep the system small enough to run efficiently on edge hardware.
- Optionally add alert/notification mechanisms for user convenience.

## Deliverables
- Code snippets and demo for deploying ML models (TensorFlow Lite) on Raspberry Pi.
- Person detection/face detection model running locally.
- System that captures and classifies images when someone appears at the door.
- (Optional) Alert mechanism — send phone notification or log event.
- (Optional) Fun feature: detect delivery personnel (pizza/package worker).
- Documentation (design doc, setup guide, final report)

## System Blocks
- **Camera Module** → Captures live video/images.
- **ML Model (TensorFlow Lite)** → Runs face/person detection on Raspberry Pi.
- **Recognition Logic** → Embedding comparison with stored trusted faces.
- **Alert Mechanism (Optional)** → Sends notification or logs recognized/unknown visitor.
- **Output Display (Optional)** → Console logs / connected device feedback.

## Hardware Requirements
- Raspberry Pi (preferably Pi 4 or Pi 5 for performance).
- Camera module compatible with Pi.
- Power supply and storage (microSD).
- (Optional) Display, buzzer, or LED for alerts.

## Software Requirements
- TensorFlow Lite for inference.
- Python (OpenCV, NumPy).
- Google Colab (for training phase).
- Linux shell/command line tools.

## Team Members & Responsibilities
- **Setup Lead** – Raspberry Pi hardware and camera configuration.: Ishrak & Armaan
- **Software Lead** – TensorFlow Lite deployment & code implementation.: Everyone
- **Networking Lead** – Alert/notification integration.: Anika
- **Writing Lead** – Documentation, project reports, GitHub maintenance. Anika & Ishrak
- **Research Lead** – Find lightweight ML models suitable for Pi.: Anika
- **Algorithm Design Lead** – Face embedding and classification pipeline.: Ishrak

## Project Timeline
| Week | Task |
|------|------|
| 1 | Repo setup, roles assigned, hardware collection |
| 2 | Learn TensorFlow Lite deployment |
| 3 | Implement face/person detection model |
| 4 | Build image capture + classification pipeline |
| 5 | Add alert mechanism + optional delivery detection |
| 6 | System integration & debugging |
| 7 | Final testing, documentation, demo video |

## References
- [MobileNets: Efficient CNNs for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- TensorFlow Lite official docs
- Raspberry Pi Camera documentation
- OpenCV face recognition tutorials
