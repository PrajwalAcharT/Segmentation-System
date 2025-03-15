# Image Segmentation API

## Overview
This project provides a Flask-based API for image segmentation using two models:
1. **SAM2 (Segment Anything Model v2)**
2. **FastSAM (Fast Segment Anything Model)**

The API automatically switches between the two models, using FastSAM for faster processing and falling back to SAM2 if needed. It supports multiple input types (images as base64, bytes, or numpy arrays) and provides mask outputs in a format compatible with SAM2.

## Features
- Dual Model Support: Switch between SAM2 and FastSAM for efficient segmentation.
- Image Input: Accepts images in base64, bytes, or numpy array formats.
- Mask Output: Returns segmentation masks with bounding boxes, areas, and stability scores.
- Configurable: Control model paths, Redis cache settings, and segmentation parameters via environment variables.
- Scalable: Utilizes a thread pool for concurrent image processing.

## Requirements
- Python 3.8+
- Redis
- CUDA (for GPU acceleration)

### Python Libraries
- Flask
- torch
- numpy
- opencv-python (cv2)
- redis
- pillow

## Installation
1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Redis is running locally or on a specified host.

## Configuration
Set the following environment variables to customize the API:

| Variable              | Description                            | Default               |
|-----------------------|----------------------------------------|-----------------------|
| `REDIS_HOST`          | Redis server hostname                  | localhost            |
| `REDIS_PORT`          | Redis server port                      | 6379                 |
| `REDIS_DB`            | Redis database index                   | 0                    |
| `REDIS_TTL`           | Cache expiration time (seconds)        | 3600                 |
| `SAM_MODEL_PATH`      | Path to the SAM2 model checkpoint      | sam2_hiera_base_plus.pt |
| `FASTSAM_MODEL_PATH`  | Path to the FastSAM model checkpoint   | FastSAM-x.pt         |
| `DEFAULT_CONF`        | Default confidence threshold           | 0.4                  |
| `DEFAULT_IOU`         | Default Intersection-over-Union (IoU)  | 0.8                  |
| `MAX_WORKERS`         | Number of worker threads               | 2                    |
| `MAX_CONTOURS`        | Maximum contours to return per mask    | 10                   |
| `MASK_POST_PROCESSING`| Enable/Disable mask post-processing    | True                 |
| `MEMORY_OPTIMIZATION` | Enable/Disable image resizing          | True                 |
| `PORT`                | Flask server port                      | 5000                 |

## Usage

1. Run the Flask API:

   ```bash
   export SAM_MODEL_PATH=/path/to/sam_model.pt
   export FASTSAM_MODEL_PATH=/path/to/fastsam_model.pt
   python app.py
   ```

2. Send an image for segmentation:

   ```bash
   curl -X POST "http://localhost:5000/segment" \
        -H "Content-Type: application/json" \
        -d '{"image": "<base64_image_string>"}'
   ```

3. Example response:

   ```json
   {
       "success": true,
       "num_masks": 3,
       "masks": [
           {
               "segmentation": [[0, 0], [100, 0], [100, 100], [0, 100]],
               "area": 10000,
               "bbox": [0, 0, 100, 100],
               "predicted_iou": 0.95,
               "stability_score": 0.9
           }
       ]
   }
   ```

## API Endpoints

### `POST /segment`

**Request:**
- `image` (string, required): Base64 encoded image or image bytes.
- `points` (optional): Array of points for segmentation guidance.
- `boxes` (optional): Array of bounding boxes for segmentation.

**Response:**
- `success` (boolean): Whether segmentation was successful.
- `num_masks` (int): Number of masks returned.
- `masks` (array): List of mask objects.

## Logging
The system uses Python's built-in `logging` module for tracking events. Logs include segmentation outcomes, errors, and cache operations.

## License
This project is released under the MIT License.

## Contributors
- Prajwal Achar T

Feel free to contribute by raising issues or submitting pull requests!

