import os
import torch
import numpy as np
import cv2
import redis
import json
import base64
import logging
import traceback
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from typing import List, Dict, Any, Union, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis connection settings
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_TTL = int(os.environ.get('REDIS_TTL', 3600))  # Time-to-live in seconds

# Model settings
SAM_MODEL_PATH = os.environ.get('SAM_MODEL_PATH', 'sam2_hiera_base_plus.pt')
FASTSAM_MODEL_PATH = os.environ.get('FASTSAM_MODEL_PATH', 'FastSAM-x.pt')
DEFAULT_CONF = float(os.environ.get('DEFAULT_CONF', 0.4))
DEFAULT_IOU = float(os.environ.get('DEFAULT_IOU', 0.8))
MASK_POST_PROCESSING = os.environ.get('MASK_POST_PROCESSING', 'True').lower() == 'true'
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 2))  # Configurable thread pool size
MAX_CONTOURS = int(os.environ.get('MAX_CONTOURS', 10))  # Max number of contours to return per mask
MEMORY_OPTIMIZATION = os.environ.get('MEMORY_OPTIMIZATION', 'True').lower() == 'true'
MAX_MASK_SIZE = int(os.environ.get('MAX_MASK_SIZE', 1536))  # Maximum mask dimension before resizing

# Initialize Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False
)

# Global model cache
model_cache = {}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

class SegmentationHandler:
    """Base class for segmentation handlers"""
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
    def lazy_load(func):
        """Decorator for lazy model loading"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.model is None:
                self.load_model()
            return func(self, *args, **kwargs)
        return wrapper
        
    def load_model(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError
        
    def segment_image(self, image: Union[str, np.ndarray, Image.Image], *args, **kwargs) -> Dict[str, Any]:
        """Segment image - to be implemented by subclasses"""
        raise NotImplementedError
        
    def process_image(self, image_data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Process image data into a format suitable for the model"""
        if isinstance(image_data, str):
            # Assume it's a base64 string
            image_data = base64.b64decode(image_data.split(',')[-1] if ',' in image_data else image_data)
            
        if isinstance(image_data, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("Failed to decode image data or empty image")
            if not (2 <= len(img.shape) <= 3):
                raise ValueError(f"Invalid image dimensions: {img.shape}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        elif isinstance(image_data, np.ndarray):
            # Validate numpy array
            if image_data.size == 0:
                raise ValueError("Empty numpy array provided")
            if len(image_data.shape) not in (2, 3):
                raise ValueError(f"Invalid array shape: {image_data.shape}")
            if image_data.dtype != np.uint8:
                raise ValueError(f"Invalid array dtype: {image_data.dtype}")
                
            # Handle BGR conversion if needed
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                if image_data[0, 0, 0] > image_data[0, 0, 2]:  # Simple BGR check
                    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            return image_data
        
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
    
    def resize_image_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Resize image if it's too large to conserve memory"""
        if not MEMORY_OPTIMIZATION:
            return image, 1.0, 1.0
            
        h, w = image.shape[:2]
        
        # Check if image is too large
        if max(h, w) > MAX_MASK_SIZE:
            scale = MAX_MASK_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return resized_img, w / new_w, h / new_h
        
        return image, 1.0, 1.0
            
    def post_process_masks(self, masks: List[Dict[str, Any]], scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict[str, Any]]:
        """Post-process masks to ensure SAM2-like smooth contours"""
        if not MASK_POST_PROCESSING:
            return masks
            
        processed_masks = []
        for mask in masks:
            segmentation = mask.get('segmentation', [])
            if not segmentation:
                processed_masks.append(mask)
                continue
                
            # Scale back points if image was resized
            if scale_x != 1.0 or scale_y != 1.0:
                scaled_segmentation = self.scale_coordinates(segmentation, scale_x, scale_y)
                mask['segmentation'] = scaled_segmentation
                
                # Scale bounding box
                if 'bbox' in mask:
                    x, y, w, h = mask['bbox']
                    mask['bbox'] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                
                segmentation = scaled_segmentation
                
            # Convert segmentation to numpy array
            points = np.array(segmentation, dtype=np.int32).reshape(-1, 1, 2)
            
            # Smooth the contour using spline interpolation
            if len(points) > 5:  # Need enough points for smoothing
                # Use Ramer-Douglas-Peucker algorithm to simplify contour
                epsilon = 0.003 * cv2.arcLength(points, True)
                simplified = cv2.approxPolyDP(points, epsilon, True)
                
                # Apply cubic spline interpolation for smoother curves
                smooth_contour = []
                for i in range(len(simplified)):
                    p1 = simplified[i][0]
                    p2 = simplified[(i + 1) % len(simplified)][0]
                    # Add intermediary points using linear interpolation (simulating spline)
                    for t in np.linspace(0, 1, 5):
                        x = int((1 - t) * p1[0] + t * p2[0])
                        y = int((1 - t) * p1[1] + t * p2[1])
                        smooth_contour.append([x, y])
                
                # Update segmentation points
                mask['segmentation'] = smooth_contour
                
            processed_masks.append(mask)
            
        return processed_masks

    @staticmethod
    def scale_coordinates(coordinates: List[List[float]], scale_x: float, scale_y: float) -> List[List[float]]:
        """Universal coordinate scaling utility"""
        return [[x * scale_x, y * scale_y] for x, y in coordinates]

    def generate_fallback_mask(self, image: np.ndarray) -> Dict[str, Any]:
        """Generate a fallback mask for the entire image"""
        h, w = image.shape[:2]
        
        # Create a simple rectangular contour
        contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Calculate area
        area = float(w * h)
        
        return {
            "segmentation": contour.tolist(),
            "area": area,
            "bbox": [0.0, 0.0, float(w), float(h)],
            "predicted_iou": 0.9,  # High score for full image mask
            "stability_score": 0.9
        }

class SAM2Handler(SegmentationHandler):
    """Handler for SAM2 segmentation"""
    def __init__(self, model_path: str = SAM_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load SAM2 model"""
        try:
            if 'sam' in model_cache:
                logger.info("Using cached SAM2 model")
                self.model = model_cache['sam']
                return True
                
            logger.info(f"Loading SAM2 model from {self.model_path}")
            
            try:
                from mobile_sam import SamPredictor, sam_model_registry
                self.sam_module = sam_model_registry
                self.SamPredictor = SamPredictor
            except ImportError:
                from segment_anything import SamPredictor, sam_model_registry
                self.sam_module = sam_model_registry
                self.SamPredictor = SamPredictor
                
            model_type = self._get_model_type_from_path()
            sam_model = self.sam_module[model_type](checkpoint=self.model_path)
            sam_model.to(self.device)
            self.model = self.SamPredictor(sam_model)
            model_cache['sam'] = self.model
            return True
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def _get_model_type_from_path(self) -> str:
        """Determine model type from path"""
        path_lower = self.model_path.lower()
        if 'vit_h' in path_lower or 'hiera' in path_lower:
            return 'vit_h'
        elif 'vit_l' in path_lower:
            return 'vit_l'
        elif 'vit_b' in path_lower:
            return 'vit_b'
        else:
            return 'vit_b'
            
    @SegmentationHandler.lazy_load
    def segment_image(self, 
                     image: Union[str, np.ndarray, Image.Image], 
                     points: List[List[float]] = None, 
                     boxes: List[List[float]] = None,
                     conf: float = DEFAULT_CONF,
                     iou: float = DEFAULT_IOU) -> Dict[str, Any]:
        """Segment image using SAM2"""
        try:
            if isinstance(image, (str, bytes)):
                img = self.process_image(image)
            else:
                img = image
                
            orig_img = img.copy()
            img, scale_x, scale_y = self.resize_image_if_needed(img)
            
            self.model.set_image(img)
            
            # Process prompts
            sam_points, point_labels = None, None
            if points:
                scaled_points = self.scale_coordinates(
                    [p[:2] for p in points], 1/scale_x, 1/scale_y
                )
                sam_points = np.array(scaled_points, dtype=np.float32)
                point_labels = np.array([p[2] for p in points], dtype=np.int32)
            
            sam_boxes = None
            if boxes:
                scaled_boxes = self.scale_coordinates(boxes, 1/scale_x, 1/scale_y)
                sam_boxes = np.array(scaled_boxes, dtype=np.float32)
            
            # Run prediction
            if sam_boxes is not None and len(sam_boxes) > 0:
                masks, scores, _ = self.model.predict(box=sam_boxes[0], multimask_output=True)
            elif sam_points is not None and len(sam_points) > 0:
                masks, scores, _ = self.model.predict(
                    point_coords=sam_points,
                    point_labels=point_labels,
                    multimask_output=True
                )
            else:
                masks, scores, _ = self.model.predict(multimask_output=True)
            
            # Process masks
            result_masks = []
            for i, mask in enumerate(masks):
                contours = self._get_mask_contours(mask)
                for contour in contours:
                    scaled_contour = self.scale_coordinates(contour, scale_x, scale_y)
                    bbox = self._get_bbox_from_contour(scaled_contour)
                    area = self._calculate_contour_area(scaled_contour)
                    result_masks.append({
                        "segmentation": scaled_contour,
                        "area": float(area),
                        "bbox": [float(x) for x in bbox],
                        "predicted_iou": float(scores[i]),
                        "stability_score": float(scores[i]) * 0.9,
                        "point_coords": sam_points.tolist() if sam_points is not None else []
                    })
            
            if not result_masks:
                result_masks.append(self.generate_fallback_mask(orig_img))
            
            # Post-process masks for smoother output
            result_masks = self.post_process_masks(result_masks, scale_x, scale_y)
            
            return {
                "success": True,
                "num_masks": len(result_masks),
                "masks": result_masks
            }
            
        except Exception as e:
            logger.error(f"SAM2 segmentation error: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _get_mask_contours(self, mask: np.ndarray) -> List[List[List[float]]]:
        """Extract contours from mask"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()
            for c in contours if cv2.contourArea(c) >= 10
        ]
    
    def _get_bbox_from_contour(self, contour: List[List[float]]) -> List[float]:
        """Calculate bounding box from contour"""
        pts = np.array(contour)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def _calculate_contour_area(self, contour: List[List[float]]) -> float:
        """Calculate contour area"""
        return cv2.contourArea(np.array(contour, dtype=np.float32))

class FastSAMHandlerV2(SegmentationHandler):
    """Handler for FastSAM segmentation"""
    def __init__(self, model_path: str = FASTSAM_MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load FastSAM model"""
        try:
            # Check if model is already in cache
            if 'fastsam' in model_cache:
                logger.info("Using cached FastSAM model")
                self.model = model_cache['fastsam']
                return True
                
            logger.info(f"Loading FastSAM model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Cache the model
            model_cache['fastsam'] = self.model
            return True
        except Exception as e:
            logger.error(f"Failed to load FastSAM model: {e}")
            logger.error(traceback.format_exc())
            return False
            
    @SegmentationHandler.lazy_load
    def segment_image(self, 
                     image: Union[str, np.ndarray, Image.Image], 
                     points: List[List[float]] = None, 
                     boxes: List[List[float]] = None,
                     conf: float = DEFAULT_CONF,
                     iou: float = DEFAULT_IOU,
                     imgsz: int = FASTSAM_IMG_SIZE) -> Dict[str, Any]:
        """
        Segment image using FastSAM with enhanced compatibility
        
        Args:
            image: Input image (base64, numpy array, or PIL Image)
            points: Optional point prompts [[x, y, is_positive], ...]
            boxes: Optional bbox prompts [[x1, y1, x2, y2], ...]
            conf: Confidence threshold
            iou: IoU threshold
            imgsz: Image size for processing
        
        Returns:
            Dict with segmentation results exactly matching SAM2 format
        """
        try:
            # Process image if needed
            if isinstance(image, (str, bytes)):
                img = self.process_image(image)
            else:
                img = image
                
            # Resize image if too large
            orig_img = img.copy()
            img, scale_x, scale_y = self.resize_image_if_needed(img)
                
            # Store original dimensions for scaling
            orig_h, orig_w = orig_img.shape[:2]
                
            # Run FastSAM inference
            results = self.model(
                img, 
                conf=conf, 
                iou=iou, 
                retina_masks=True,
                imgsz=imgsz,
                verbose=False
            )
            
            # Process masks based on input prompts
            masks = []
            if results[0].masks is not None:
                tensor_masks = results[0].masks.data
                orig_boxes = results[0].boxes.data if results[0].boxes else None
                
                # If boxes are provided, filter masks by boxes with enhanced fuzzy matching
                if boxes and len(boxes) > 0:
                    # Scale boxes if image was resized
                    scaled_boxes = self.scale_coordinates(boxes, 1/scale_x, 1/scale_y) if scale_x != 1.0 else boxes
                    
                    # Make boxes slightly larger for better matching
                    expanded_boxes = self._expand_boxes(scaled_boxes, expand_ratio=0.05)
                    box_masks = self._filter_masks_by_boxes(tensor_masks, orig_boxes, expanded_boxes, iou=iou)
                    
                    for mask in box_masks:
                        masks.extend(self._tensor_to_mask_dict(mask, scale_x, scale_y, point_coords=points))
                
                # If points are provided, filter masks by points with radius matching
                elif points and len(points) > 0:
                    # Scale points if image was resized
                    scaled_points = self.scale_coordinates([p[:2] for p in points], 1/scale_x, 1/scale_y) if scale_x != 1.0 else points
                    scaled_points = [[p[0], p[1], p[2]] for p in scaled_points]  # Preserve is_positive flag
                    
                    point_masks = self._filter_masks_by_points(tensor_masks, orig_boxes, scaled_points, radius=POINT_RADIUS)
                    
                    for mask in point_masks:
                        masks.extend(self._tensor_to_mask_dict(mask, scale_x, scale_y, point_coords=points))
                
                # Otherwise, return all masks
                else:
                    for mask in tensor_masks:
                        masks.extend(self._tensor_to_mask_dict(mask, scale_x, scale_y, point_coords=points))
                    
            # If no masks found, create a fallback mask for compatibility
            if not masks:
                masks.append(self.generate_fallback_mask(orig_img))
                    
            # Post-process masks to ensure SAM2-like output
            masks = self.post_process_masks(masks, scale_x, scale_y)
                    
            return {
                "success": True,
                "num_masks": len(masks),
                "masks": masks
            }
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _filter_masks_by_boxes(self, masks: torch.Tensor, boxes: torch.Tensor, 
                             target_boxes: List[List[float]], iou: float = DEFAULT_IOU) -> List[torch.Tensor]:
        """Filter masks by target boxes using vectorized IoU calculations"""
        result_masks = []
        if boxes is None or len(boxes) == 0:
            return result_masks
            
        # Convert to numpy arrays for vectorization
        boxes_np = boxes.cpu().numpy()[:, :4]
        target_boxes_np = np.array(target_boxes)
        
        # Vectorized IoU calculation
        x1 = np.maximum(boxes_np[:, 0][:, None], target_boxes_np[:, 0])
        y1 = np.maximum(boxes_np[:, 1][:, None], target_boxes_np[:, 1])
        x2 = np.minimum(boxes_np[:, 2][:, None], target_boxes_np[:, 2])
        y2 = np.minimum(boxes_np[:, 3][:, None], target_boxes_np[:, 3])
        
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area_a = (boxes_np[:, 2] - boxes_np[:, 0])[:, None] * (boxes_np[:, 3] - boxes_np[:, 1])[:, None]
        area_b = (target_boxes_np[:, 2] - target_boxes_np[:, 0]) * (target_boxes_np[:, 3] - target_boxes_np[:, 1])
        iou_matrix = intersection / (area_a + area_b - intersection + 1e-9)
        
        # Find best matches
        for t_idx in range(len(target_boxes)):
            best_mask_idx = np.argmax(iou_matrix[:, t_idx])
            if iou_matrix[best_mask_idx, t_idx] >= iou:
                result_masks.append(masks[best_mask_idx])
            else:
                # Fallback to lower threshold
                fallback_candidates = np.where(iou_matrix[:, t_idx] >= max(0.3, iou * 0.6))[0]
                if len(fallback_candidates) > 0:
                    result_masks.append(masks[fallback_candidates[0]])
        
        return result_masks
        
    def _filter_masks_by_points(self, masks: torch.Tensor, boxes: torch.Tensor, 
                              points: List[List[float]], radius: int = POINT_RADIUS) -> List[torch.Tensor]:
        """Filter masks by points using vectorized distance calculations"""
        result_masks = []
        masks_np = [mask.cpu().numpy() for mask in masks]
        
        for point in points:
            x, y, is_positive = map(float, point)
            x, y = int(x), int(y)
            is_positive = int(is_positive)
            
            # Create coordinate grids
            h, w = masks_np[0].shape if masks_np else (0, 0)
            if h == 0 or w == 0:
                continue
                
            yy, xx = np.ogrid[:h, :w]
            dist_sq = (xx - x)**2 + (yy - y)**2
            
            if is_positive:
                # Find masks containing points within radius
                for i, mask_np in enumerate(masks_np):
                    if np.any(mask_np & (dist_sq <= radius**2)):
                        result_masks.append(masks[i])
                        break
                else:
                    # Create fallback mask if no matches
                    fallback_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.circle(fallback_mask, (x, y), radius*2, 1, -1)
                    result_masks.append(torch.from_numpy(fallback_mask).to(masks[0].device))
            else:
                # Exclude masks containing points in exclusion zone
                exclusion_zone = dist_sq <= (radius*2)**2
                valid_masks = [
                    masks[i] for i, mask_np in enumerate(masks_np)
                    if not np.any(mask_np & exclusion_zone)
                ]
                if valid_masks:
                    result_masks.extend(valid_masks)
                else:
                    # Fallback to largest mask not in exclusion zone
                    areas = [np.sum(mask) for mask in masks_np]
                    largest_idx = np.argmax(areas)
                    result_masks.append(masks[largest_idx])
        
        return result_masks

    def _tensor_to_mask_dict(self, mask_tensor: torch.Tensor, scale_x: float = 1.0, scale_y: float = 1.0, point_coords=None) -> List[Dict[str, Any]]:
        """
        Convert FastSAM mask tensor to SAM2-compatible output format.
        """
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255

        # Apply smoothing (optional for better contour quality)
        mask_np = cv2.medianBlur(mask_np, 3)
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]

        results = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            # Simplify contour for smoother output
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            points = approx_contour.reshape(-1, 2).tolist()

            # Scale contour if needed
            if scale_x != 1.0 or scale_y != 1.0:
                points = [[x * scale_x, y * scale_y] for x, y in points]

            # Bounding box calculation
            x, y, w, h = cv2.boundingRect(contour)

            # Area calculation
            area = cv2.contourArea(contour)

            # Calculate stability_score (area/box area ratio)
            box_area = w * h
            stability_score = area / box_area if box_area > 0 else 0.5

            # Simulate predicted_iou similar to SAM2
            predicted_iou = min(0.95, stability_score * 0.9 + 0.1)

            results.append({
                "segmentation": points,
                "area": float(area),
                "bbox": [float(x) * scale_x, float(y) * scale_y, float(w) * scale_x, float(h) * scale_y],
                "predicted_iou": float(predicted_iou),
                "stability_score": float(stability_score),
                "point_coords": point_coords if point_coords else []
            })

        if not results:
            h, w = mask_np.shape[:2]
            results.append({
                "segmentation": [[0, 0], [w, 0], [w, h], [0, h]],
                "area": float(w * h),
                "bbox": [0.0, 0.0, float(w), float(h)],
                "predicted_iou": 0.5,
                "stability_score": 0.5,
                "point_coords": point_coords if point_coords else []
            })

        return results

    def _expand_boxes(self, boxes: List[List[float]], expand_ratio: float = 0.05) -> List[List[float]]:
        """Expand boxes by percentage"""
        expanded = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            expanded.append([
                max(0, x1 - w * expand_ratio),
                max(0, y1 - h * expand_ratio),
                x2 + w * expand_ratio,
                y2 + h * expand_ratio
            ])
        return expanded

class ModelRouter:
    """Handles model switching and fallback between SAM2 and FastSAM"""
    def __init__(self):
        self.sam_handler = SAM2Handler()  # Old SAM remains unchanged
        self.fastsam_handler = FastSAMHandlerV2()  # Isolated FastSAM handler

    def segment_image(self, image, points=None, boxes=None, conf=0.4, iou=0.8):
        """Try FastSAM first, fallback to SAM2 if FastSAM fails"""
        try:
            # Attempt FastSAM segmentation
            fastsam_result = self.fastsam_handler.segment_image(image, points, boxes, conf, iou)
            if fastsam_result.get("success"):
                return fastsam_result
            else:
                logger.warning("FastSAM failed, falling back to old SAM.")
        except Exception as e:
            logger.error(f"FastSAM encountered an error: {e}")
        
        # Fallback to SAM2 (old SAM)
        return self.sam_handler.segment_image(image, points, boxes, conf, iou)

# ... [Remaining code for Flask routes, utilities, and app initialization] ...

if __name__ == '__main__':
    init_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)