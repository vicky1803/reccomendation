# StyGig - Fashion Recommendation Engine

## User Analysis Module

A computer vision-based feature extraction system for fashion recommendations, analyzing body shape and skin tone from images.

## Features

- **Body Shape Classification**: Classifies body shapes into Inverted Triangle, Pear, Hourglass, or Rectangle using MediaPipe pose estimation
- **Skin Tone Analysis**: Determines warm/cool undertones using CIELAB color space and face mesh segmentation
- **Fashion-Tuned Metrics**: Calculates shoulder-to-hip ratios and waist measurements specifically for fashion applications

## Installation

```bash
pip install -r requirements.txt
```

**Note:** The system will automatically download the required MediaPipe models (~50MB total) on first run. They will be cached in the `models/` directory.

## Project Structure

```
├── src/
│   └── analysis/
│       ├── __init__.py
│       ├── extractor.py      # Pose & metric extraction
│       ├── skin_tone.py      # Skin tone analysis
│       └── classifier.py     # Body shape classification
├── local_testing/
│   └── visualize_features.py # Visualization tool
└── requirements.txt
```

## Usage

### Basic Usage

```python
from src.analysis import PoseExtractor, SkinToneAnalyzer, BodyShapeClassifier
import cv2

# Load image
image = cv2.imread('path/to/image.jpg')

# Extract body metrics
pose_extractor = PoseExtractor()
metrics = pose_extractor.extract_metrics(image)

# Classify body shape
classifier = BodyShapeClassifier()
body_shape = classifier.classify(metrics)

# Analyze skin tone
skin_analyzer = SkinToneAnalyzer()
skin_tone = skin_analyzer.get_skin_tone(image)

print(f"Body Shape: {body_shape}")
print(f"Skin Undertone: {skin_tone['undertone']}")
```

### Visualization Tool

To visualize the analysis results:

```bash
python local_testing/visualize_features.py path/to/image.jpg
```

This will display:
- Neon green lines showing shoulder and hip measurements
- Red line showing estimated waistline
- Text overlay with body shape and skin undertone classification

## Technical Details

### Body Measurements

- **Shoulders**: Distance between landmarks 11 (left) and 12 (right)
- **Hips**: Distance between landmarks 23 (left) and 24 (right)
- **Waist**: Estimated at 15% above hip center from the shoulder-hip midpoint

### Body Shape Classification Rules

- **Inverted Triangle**: Shoulder-to-hip ratio > 1.05
- **Pear**: Shoulder-to-hip ratio < 0.95
- **Hourglass**: 0.95 ≤ ratio ≤ 1.05 AND waist < 80% of hips
- **Rectangle**: 0.95 ≤ ratio ≤ 1.05 AND waist ≥ 80% of hips

### Skin Tone Analysis

- Uses CIELAB color space (perceptually uniform)
- Excludes eyes and lips from analysis
- K-Means clustering (k=2) to find dominant skin color
- **Warm**: b-channel > 0 (yellow undertone)
- **Cool**: b-channel ≤ 0 (blue undertone)

## Dependencies

- `mediapipe>=0.10.30` - Pose estimation and face mesh (uses new Tasks API)
- `opencv-python>=4.8.0` - Image processing
- `scikit-learn>=1.3.0` - K-Means clustering
- `matplotlib>=3.8.0` - Visualization
- `numpy>=1.24.0` - Numerical operations

## Configuration

MediaPipe is configured with:
- **Pose Landmarker**: Heavy model for high accuracy, running in IMAGE mode
- **Face Landmarker**: Face mesh with refined landmarks
- Models are automatically downloaded from Google Cloud Storage on first use

## License

See [LICENSE](LICENSE) file for details.
