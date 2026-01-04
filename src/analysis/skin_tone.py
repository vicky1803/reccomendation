"""
Skin Tone Analysis Module
Analyzes skin tone undertones for fashion color recommendations
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans
from typing import Dict, Optional, Tuple, Any
import urllib.request
import os


class SkinToneAnalyzer:
    """
    Analyzes skin tone using CIELAB color space to determine warm/cool undertones
    for fashion color recommendations.
    """
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "models/face_landmarker.task"
    
    def __init__(self) -> None:
        """Initialize MediaPipe Face Mesh for face segmentation."""
        # Download model if not exists
        self._ensure_model_downloaded()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
    
    def _ensure_model_downloaded(self) -> None:
        """Download the face landmarker model if it doesn't exist."""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(os.path.dirname(self.MODEL_PATH) or "models", exist_ok=True)
            print(f"Downloading face detection model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")
        
        # Define landmark indices to exclude (eyes and lips)
        # Eyes: indices around the eye regions
        self.eye_indices = set([
            # Left eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Right eye
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ])
        
        # Lips: indices around the mouth region
        self.lip_indices = set([
            # Outer lips
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            # Inner lips
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            # Additional mouth area
            185, 40, 39, 37, 0, 267, 269, 270, 409
        ])
    
    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def _create_face_mask(
        self, 
        landmarks: Any, 
        image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Create a binary mask for the face, excluding eyes and lips.
        
        Args:
            landmarks: MediaPipe face mesh landmarks (list of NormalizedLandmark)
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            Binary mask where face pixels (excluding eyes/lips) are 255, others 0
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get all face landmark points
        face_points = []
        for idx, landmark in enumerate(landmarks):
            # Skip eye and lip landmarks
            if idx in self.eye_indices or idx in self.lip_indices:
                continue
            
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            face_points.append([x, y])
        
        # Create convex hull from remaining points
        if len(face_points) > 0:
            face_points = np.array(face_points, dtype=np.int32)
            hull = cv2.convexHull(face_points)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Additionally exclude specific eye and lip regions with circular masks
            for idx in self.eye_indices | self.lip_indices:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    # Draw a small circle to exclude this area (dynamic radius for different image sizes)
                    radius = max(1, int(width * 0.005))
                    cv2.circle(mask, (x, y), radius, 0, -1)
        
        return mask
    
    def _extract_skin_pixels(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract skin pixels from the image using the mask.
        
        Args:
            image: Input image in BGR format
            mask: Binary mask of face region
            
        Returns:
            Array of skin pixels in BGR format, or None if no pixels found
        """
        # Apply mask to extract only face pixels
        skin_pixels = image[mask > 0]
        
        if len(skin_pixels) == 0:
            return None
        
        return skin_pixels
    
    def _get_dominant_color(
        self, 
        pixels: np.ndarray, 
        n_clusters: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use K-Means clustering to find the dominant skin color.
        
        Args:
            pixels: Array of pixel values
            n_clusters: Number of clusters for K-Means
            
        Returns:
            Tuple of (dominant_color, cluster_centers)
        """
        # Reshape pixels for K-Means
        pixels_reshaped = pixels.reshape(-1, 3)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_reshaped)
        
        # Get the cluster with the most pixels (dominant color)
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        dominant_cluster = np.argmax(label_counts)
        
        dominant_color = kmeans.cluster_centers_[dominant_cluster]
        
        return dominant_color, kmeans.cluster_centers_
    
    def _classify_undertone(
        self, 
        lab_color: np.ndarray, 
        threshold: float = 128.0
    ) -> str:
        """
        Classify skin undertone as Warm or Cool based on LAB b-channel.
        
        Args:
            lab_color: Color in LAB format [L, A, B]
            threshold: Threshold for classification (default: 128.0)
            
        Returns:
            "Warm" or "Cool" classification
        """
        # In LAB color space:
        # OpenCV offsets LAB channels: L=[0,255], A=[0,255], B=[0,255]
        # Neutral point is 128. B > 128 = yellow (warm), B < 128 = blue (cool)
        b_channel = lab_color[2]
        
        if b_channel > threshold:
            return "Warm"
        else:
            return "Cool"
    
    def get_skin_tone(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze skin tone from an image and determine warm/cool undertone.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary containing:
                - lab_values: Dominant LAB color values [L, A, B]
                - undertone: "Warm" or "Cool" classification
                - bgr_values: Corresponding BGR values
                - confidence: Measure of color clustering quality
            Returns None if face detection fails
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image
        results = self.face_mesh.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None
        
        # Use the first detected face
        face_landmarks = results.face_landmarks[0]
        
        # Create face mask excluding eyes and lips
        mask = self._create_face_mask(face_landmarks, image.shape)
        
        # Extract skin pixels
        skin_pixels = self._extract_skin_pixels(image, mask)
        
        if skin_pixels is None or len(skin_pixels) < 100:
            return None
        
        # Get dominant color using K-Means clustering
        dominant_bgr, cluster_centers = self._get_dominant_color(skin_pixels, n_clusters=2)
        
        # Convert BGR to LAB color space
        # Create a 1x1 image with the dominant color
        dominant_bgr_image = dominant_bgr.reshape(1, 1, 3).astype(np.uint8)
        dominant_lab_image = cv2.cvtColor(dominant_bgr_image, cv2.COLOR_BGR2LAB)
        dominant_lab = dominant_lab_image[0, 0]
        
        # Classify undertone
        undertone = self._classify_undertone(dominant_lab)
        
        # Calculate confidence based on cluster separation
        # Higher separation means more confident classification
        cluster_distance = np.linalg.norm(cluster_centers[0] - cluster_centers[1])
        confidence = min(cluster_distance / 100.0, 1.0)  # Normalize to 0-1
        
        return {
            'lab_values': {
                'L': float(dominant_lab[0]),
                'A': float(dominant_lab[1]),
                'B': float(dominant_lab[2])
            },
            'undertone': undertone,
            'bgr_values': {
                'B': float(dominant_bgr[0]),
                'G': float(dominant_bgr[1]),
                'R': float(dominant_bgr[2])
            },
            'confidence': float(confidence)
        }
