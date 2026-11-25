# AI-Based Industrial PCB Defect Detection

An AI-powered Android application that automatically detects and assesses defects in Printed Circuit Boards (PCBs) using deep learning.

## Project Overview

Manual PCB inspection can be time-consuming and may lead to human errors. This project uses a YOLO-based deep learning model to identify PCB defects from images and an on-device Llama AI model to generate defect explanations and inspection recommendations.

The application is developed using Flutter and runs on Android devices.

## Features

- Select PCB images using the camera or gallery.
- Detect PCB defects using a YOLO-based model.
- Identify six types of PCB defects.
- Display detection confidence and defect severity.
- Generate AI-based inspection reports.
- View previous inspection results and history.
- Perform AI inference directly on the Android device.

## Defect Categories

The system detects the following six defect types:

1. Open
2. Short
3. Mousebite
4. Spur
5. Spurious Copper
6. Pin Hole

## Technologies Used

| Component | Technology |
|---|---|
| Mobile application | Flutter |
| Programming languages | Dart, Kotlin |
| Defect detection | YOLOv8 |
| Detection model format | ONNX |
| AI report generation | Llama 3.2 1B |
| On-device AI inference | ExecuTorch |
| Image processing | Image package |
| Development environment | Android Studio, VS Code |

## Model Files

The application uses the following model assets:

- `assets/models/deeppcb_best.onnx` – PCB defect detection model.
- `assets/models/llama/tokenizer.json` – Llama tokenizer.

**Note:** The large Llama `.pte` model file is excluded from GitHub because it exceeds GitHub's individual file size limit. It must be provided locally for the on-device Llama report-generation feature to work.

## Installation and Setup

### Prerequisites

- Flutter SDK
- Android Studio or VS Code
- Android device or emulator
- Git

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/pavankalyanbt20-hue/Printed-circuit-board-inspection.git
   ```

2. Open the project directory:

   ```bash
   cd Printed-circuit-board-inspection
   ```

3. Install dependencies:

   ```bash
   flutter pub get
   ```

4. Ensure the required model files are available in their respective asset directories.

5. Connect an Android device or start an emulator.

6. Run the application:

   ```bash
   flutter run
   ```

## Model Performance

The YOLO-based PCB defect detection model achieved the following validation results:

| Metric | Result |
|---|---:|
| Precision | 94.08% |
| Recall | 89.01% |
| mAP@0.50 | 92.75% |
| mAP@0.50:0.95 | 49.07% |

An additional ONNX evaluation on 500 images reported an F1-score of 91.53%.

*Note: These results come from different evaluation procedures and should not be directly compared.*

## Project Structure

```text
Printed-circuit-board-inspection/
│
├── android/
│   └── app/
│
├── assets/
│   └── models/
│       ├── deeppcb_best.onnx
│       └── llama/
│           └── tokenizer.json
│
├── lib/
│   └── main.dart
│
├── PCB Dataset/
│
├── pubspec.yaml
├── analysis_options.yaml
└── README.md
```

## Future Enhancements

- Improve detection accuracy across different PCB designs.
- Optimize on-device inference speed and memory usage.
- Improve AI-generated inspection reports.
- Add more PCB defect categories.
- Evaluate performance across additional Android devices.

## Project Team

- **Pavan Kalyan**
- **Pradeem Gani**

## License

This project is developed for educational and academic purposes.
