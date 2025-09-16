"""
Image Processing Utilities
Common image processing functions for the visual intelligence pipeline
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any
import logging
from dataclasses import dataclass

from src.shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class ImageMetadata:
    """Metadata structure for processed images"""
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    quality_score: float
    brightness: float
    contrast: float
    sharpness: float

class ImageProcessor:
    """
    Comprehensive image processing utilities for street scene analysis
    """
    
    @staticmethod
    def load_image_from_uri(image_uri: str) -> Optional[np.ndarray]:
        """Load image from various URI formats (GCS, local, base64)"""
        try:
            if image_uri.startswith('gs://'):
                return ImageProcessor._load_from_gcs(image_uri)
            elif image_uri.startswith('data:image'):
                return ImageProcessor._load_from_base64(image_uri)
            elif image_uri.startswith('http'):
                return ImageProcessor._load_from_url(image_uri)
            else:
                return ImageProcessor._load_from_local(image_uri)
        except Exception as e:
            logger.error(f"Failed to load image from {image_uri}: {e}")
            return None
    
    @staticmethod
    def _load_from_gcs(gcs_uri: str) -> np.ndarray:
        """Load image from Google Cloud Storage"""
        from google.cloud import storage
        
        # Parse GCS URI
        bucket_name = gcs_uri.split('/')[2]
        blob_name = '/'.join(gcs_uri.split('/')[3:])
        
        # Download image
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        image_bytes = blob.download_as_bytes()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    @staticmethod
    def _load_from_base64(data_uri: str) -> np.ndarray:
        """Load image from base64 data URI"""
        # Extract base64 data
        header, data = data_uri.split(',', 1)
        image_bytes = base64.b64decode(data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    @staticmethod
    def _load_from_url(url: str) -> np.ndarray:
        """Load image from HTTP URL"""
        import requests
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    @staticmethod
    def _load_from_local(file_path: str) -> np.ndarray:
        """Load image from local file system"""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        return image
    
    @staticmethod
    def resize_image(
        image: np.ndarray, 
        target_size: Tuple[int, int], 
        maintain_aspect_ratio: bool = True
    ) -> np.ndarray:
        """Resize image to target size"""
        if not maintain_aspect_ratio:
            return cv2.resize(image, target_size)
        
        # Calculate aspect ratio preserving resize
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size if needed
        if new_w != target_w or new_h != target_h:
            # Create padded image
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            
            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        
        return resized
    
    @staticmethod
    def normalize_image(image: np.ndarray, method: str = 'imagenet') -> np.ndarray:
        """Normalize image using different methods"""
        if method == 'imagenet':
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            normalized = (normalized - mean) / std
            
        elif method == 'zero_one':
            # Simple [0, 1] normalization
            normalized = image.astype(np.float32) / 255.0
            
        elif method == 'minus_one_one':
            # [-1, 1] normalization
            normalized = (image.astype(np.float32) / 127.5) - 1.0
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better analysis"""
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (subtle sharpening)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    @staticmethod
    def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
        """Calculate various image quality metrics"""
        # Convert to grayscale for some calculations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness (mean intensity)
        brightness = np.mean(gray) / 255.0
        
        # Contrast (standard deviation of intensity)
        contrast = np.std(gray) / 255.0
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var() / 10000.0  # Normalized
        
        # Noise estimation (using high-frequency content)
        noise = ImageProcessor._estimate_noise(gray)
        
        # Overall quality score (weighted combination)
        quality_score = (
            0.3 * min(brightness * 2, 1.0) +  # Prefer moderate brightness
            0.3 * min(contrast * 2, 1.0) +    # Prefer good contrast
            0.3 * min(sharpness, 1.0) +       # Prefer sharp images
            0.1 * max(0, 1.0 - noise)         # Penalize noise
        )
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'noise': noise,
            'quality_score': quality_score
        }
    
    @staticmethod
    def _estimate_noise(gray_image: np.ndarray) -> float:
        """Estimate noise level in grayscale image"""
        # Use Laplacian to detect edges/noise
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        
        # Calculate noise as variance of high-frequency components
        noise_level = np.var(laplacian) / (gray_image.shape[0] * gray_image.shape[1])
        
        # Normalize to [0, 1] range
        return min(noise_level / 1000.0, 1.0)
    
    @staticmethod
    def extract_image_metadata(image: np.ndarray, original_bytes: Optional[bytes] = None) -> ImageMetadata:
        """Extract comprehensive metadata from image"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Calculate size
        size_bytes = image.nbytes if original_bytes is None else len(original_bytes)
        
        # Determine format (simplified)
        format_type = "BGR" if channels == 3 else "GRAY"
        
        # Calculate quality metrics
        quality_metrics = ImageProcessor.calculate_image_quality_metrics(image)
        
        return ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            format=format_type,
            size_bytes=size_bytes,
            quality_score=quality_metrics['quality_score'],
            brightness=quality_metrics['brightness'],
            contrast=quality_metrics['contrast'],
            sharpness=quality_metrics['sharpness']
        )
    
    @staticmethod
    def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
        """Create thumbnail of image"""
        return ImageProcessor.resize_image(image, size, maintain_aspect_ratio=True)
    
    @staticmethod
    def apply_roi_mask(image: np.ndarray, roi_polygon: List[Tuple[int, int]]) -> np.ndarray:
        """Apply region of interest mask to image"""
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Fill polygon
        roi_array = np.array(roi_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [roi_array], 255)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image
    
    @staticmethod
    def detect_motion_blur(image: np.ndarray) -> float:
        """Detect motion blur in image using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize blur score (lower variance = more blur)
        blur_threshold = 100.0
        blur_score = min(laplacian_var / blur_threshold, 1.0)
        
        return 1.0 - blur_score  # Return blur level (0 = no blur, 1 = heavy blur)
    
    @staticmethod
    def correct_perspective(
        image: np.ndarray, 
        src_points: List[Tuple[int, int]], 
        dst_size: Tuple[int, int]
    ) -> np.ndarray:
        """Correct perspective distortion in image"""
        # Define source and destination points
        src_pts = np.float32(src_points)
        dst_pts = np.float32([
            [0, 0],
            [dst_size[0], 0],
            [dst_size[0], dst_size[1]],
            [0, dst_size[1]]
        ])
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transformation
        corrected = cv2.warpPerspective(image, matrix, dst_size)
        
        return corrected
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format: str = 'jpg', quality: int = 85) -> str:
        """Encode image to base64 string"""
        # Encode image
        if format.lower() in ['jpg', 'jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            mime_type = 'image/jpeg'
        elif format.lower() == 'png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            _, buffer = cv2.imencode('.png', image, encode_param)
            mime_type = 'image/png'
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Convert to base64
        encoded = base64.b64encode(buffer).decode('utf-8')
        
        # Create data URI
        data_uri = f"data:{mime_type};base64,{encoded}"
        
        return data_uri
    
    @staticmethod
    def batch_process_images(
        image_uris: List[str], 
        processing_function: callable,
        max_workers: int = 4
    ) -> List[Any]:
        """Process multiple images in parallel"""
        import concurrent.futures
        
        def process_single_image(uri):
            try:
                image = ImageProcessor.load_image_from_uri(uri)
                if image is not None:
                    return processing_function(image)
                return None
            except Exception as e:
                logger.error(f"Error processing image {uri}: {e}")
                return None
        
        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_image, image_uris))
        
        return results
