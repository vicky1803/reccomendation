StyGig MVP: Complete Implementation Guide

## 1. Project Overview & Architecture
**Goal:** Build a functional MVP for "StyGig," an AI fashion recommendation system with Virtual Try-On (VTON).
**Core Logic:** 1. **Analyze User:** Extract body landmarks (MediaPipe) and skin tone (K-Means). Classify body shape using a lightweight ML model (XGBoost).
2. **Recommend:** Use OpenCLIP to find visually matching clothes from an inventory, then filter them based on body shape/skin tone compatibility rules.
3. **Visualize:** Apply the recommended garment to the user's image using IDM-VTON.

## 2. Tech Stack & Dependencies
**Language:** Python 3.10+
**Key Libraries:**
* **Computer Vision:** `mediapipe`, `opencv-python`, `pillow`
* **ML & Data:** `numpy`, `pandas`, `scikit-learn`, `xgboost`
* **Deep Learning:** `torch`, `transformers` (for CLIP), `diffusers` (for VTON), `accelerate`

### Action Item: `requirements.txt`
Create a `requirements.txt` file with the following:
```text
mediapipe
opencv-python
numpy
pandas
xgboost
scikit-learn
transformers
torch
torchvision
pillow
accelerate
diffusers
huggingface_hub
scipy
```

Phase 1: Data Layer & "Zero-Shot" Indexing
Context: We do not train a recommendation model from scratch. We "index" a clothing inventory using OpenCLIP embeddings so we can search it semantically.

Step 1.1: Setup Inventory
Directory: data/inventory/images/

Action: Ensure this folder contains 10-20 sample clothing images (download simple images like "red_dress.jpg", "blue_shirt.jpg" manually for testing).

Step 1.2: The Indexer Script
File: src/data/indexer.py

Logic:

Load CLIPModel and CLIPProcessor (pretrained: openai/clip-vit-base-patch32).

Iterate through data/inventory/images/.

For each image:

Preprocess and pass to model to get Image Embeddings.

Normalize the vector (L2 norm) for cosine similarity.

Save the matrix of embeddings to data/embeddings.npy.

Save a corresponding data/metadata.csv containing columns: [filename, category, tags].

Phase 2: User Analysis Module
Context: We need to extract structured data from the raw user photo.

Step 2.1: Landmark Extraction
File: src/analysis/extractor.py

Logic:

Initialize mediapipe.solutions.pose.

Function get_landmarks(image_path):

Run inference.

Extract critical points: Shoulders (11,12), Hips (23,24), Waist (estimate midpoint), Knees (25,26).

Return dictionary of coordinates.

Step 2.2: Body Shape Classification (XGBoost)
File: scripts/train_shape_model.py (One-time setup)

Context: Since we don't have labeled user data, we synthesize it to train the classifier.

Logic:

Generate 1,000 rows of synthetic data: shoulder_width, waist_width, hip_width.

Labeling Rules (Ground Truth):

If (Shoulder > Hip * 1.05) → Label: "Inverted Triangle"

If (Hip > Shoulder * 1.05) → Label: "Pear"

If (Waist < Shoulder * 0.75 AND Waist < Hip * 0.75) → Label: "Hourglass"

Else → Label: "Rectangle"

Train an XGBoost Classifier on this data.

Save model to models/body_shape_xgb.json.

File: src/analysis/classifier.py (Inference)

Logic:

Load models/body_shape_xgb.json.

Take landmarks from Step 2.1, calculate widths, and predict shape.

Step 2.3: Skin Tone Extraction
File: src/analysis/skin_tone.py

Logic:

Use MediaPipe Face Mesh to generate a binary mask for the face (excluding eyes/lips).

Extract pixels from the face region.

Convert BGR pixels to CIELAB color space.

Run K-Means (k=1) to find the dominant color.

Mapper: Define logic to map the CIELAB values to seasonality:

High L (Light) + Warm (b+) → "Spring"

Low L (Dark) + Cool (b-) → "Winter" (simplified logic for MVP).


Phase 3: Recommendation Engine
Context: Combine the visual search (CLIP) with the compatibility rules (Analysis).

Step 3.1: The Engine
File: src/recommendation/engine.py

Logic:

Load: Load embeddings.npy and metadata.csv.

Function: recommend(user_analysis, query_text="casual outfit")

Step A (Visual Search):

Convert query_text to CLIP text embedding.

Calculate Cosine Similarity against all item embeddings.

Select Top 20 candidates.

Step B (Rule Filtering):

Apply fashion rules:

If User="Pear": Penalize items tagged "skinny jeans".

If User="Winter": Boost items with colors ["blue", "black", "white"].

Return: Top 3 filenames.

Phase 4: Virtual Try-On (VTON)
Context: The "Wow" factor. Using IDM-VTON.

File: src/vton/inference.py

Logic:

Function: try_on(user_image_path, garment_image_path)

Pipeline:

Load yisol/IDM-VTON pipeline from Diffusers.

(Crucial: If no GPU is detected, mock this step by simply placing the garment image next to the user image and print "GPU required for VTON" to avoid crashing).

If GPU exists: Run inference with standard parameters.

Save output to output/tryon_result.jpg.

7. Phase 5: Integration (The Main Loop)
File: main.py Logic:

Python

def main():
    # 1. Input
    user_img = "test_user.jpg"
    
    # 2. Analysis
    landmarks = extractor.get_landmarks(user_img)
    shape = classifier.predict_shape(landmarks)
    skin_tone = skin_analyzer.get_tone(user_img)
    print(f"User is {shape} with {skin_tone} skin.")
    
    # 3. Recommendation
    recs = recommender.recommend(shape, skin_tone, query="party wear")
    best_item = recs[0]
    print(f"Recommending: {best_item}")
    
    # 4. VTON
    result_path = vton.try_on(user_img, best_item)
    print(f"Result saved to {result_path}")