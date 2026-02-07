#!/usr/bin/env python3
"""
Clean Single Board Images Module
Uses board0.png as ideal reference image to find and normalize boards in other images.
- Copies board0.png to cleaned directory as-is (best image, no margins)
- Detects boards using multi-method approach:
  1. Feature matching (AKAZE/ORB) - handles rotation and scale
  2. Shape matching (contour-based) - rotation-invariant
  3. Edge detection - fallback for difficult cases
- Uses board0 as reference to find the 4 corners of the detected board
- Applies perspective transformation using OpenCV to normalize board orientation
- Crops detected board regions to normalized boundaries

Usage:
    python clean_source_boards.py
    python clean_source_boards.py --source training_data --output training_output/cleaned
"""
import sys
from pathlib import Path

# Add parent directory to path so config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import shutil
from config import CLEANED_DIR, REFERENCE_IMAGE


class SourceBoardCleaner:
    """Clean single board images using template matching with reference image"""
    
    def __init__(self, source_dir: str = "training_data", 
                 output_dir: str = None,
                 reference_image: str = None):
        """
        Initialize source board cleaner.
        
        Args:
            source_dir: Source directory containing images (default: from config)
            output_dir: Where to save cleaned images (default: from config)
            reference_image: Reference image name (default: from config)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir or CLEANED_DIR)
        self.reference_image = reference_image or REFERENCE_IMAGE
        self.reference_img = None
        self.reference_shape = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate source directory
        if not self.source_dir.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")
        
        # Load reference image
        self._load_reference_image()
        
        self.stats = {
            'copied': 0,
            'matched': 0,
            'failed': 0,
            'errors': []
        }
    
    def _load_reference_image(self):
        """Load the reference image for template matching"""
        ref_path = self.source_dir / self.reference_image
        
        if not ref_path.exists():
            raise ValueError(f"Reference image not found: {ref_path}")
        
        self.reference_img = cv2.imread(str(ref_path))
        if self.reference_img is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        
        self.reference_shape = self.reference_img.shape
        print(f"✓ Loaded reference image: {self.reference_image}")
        print(f"  Dimensions: {self.reference_shape[1]}×{self.reference_shape[0]}")
    
    def copy_reference_image(self):
        """Copy reference image to output directory as-is"""
        src = self.source_dir / self.reference_image
        dst = self.output_dir / self.reference_image
        
        shutil.copy2(str(src), str(dst))
        self.stats['copied'] += 1
        print(f"\n▶ {self.reference_image}", end=" ")
        print(f"✓ [COPIED - NO MARGINS] {self.reference_shape[1]}×{self.reference_shape[0]}")
    
    def correct_perspective_and_rotation(self, image: np.ndarray, reference_img: np.ndarray, 
                                        detected_bbox) -> np.ndarray:
        """
        Use board0 reference to find corners in the detected board and apply perspective transformation.
        
        Args:
            image: Input image
            reference_img: Reference image (board0.png) for corner detection
            detected_bbox: (x, y, w, h) bounding box of board
            
        Returns:
            Perspective-transformed image, or original crop if transformation fails
        """
        x, y, w, h = detected_bbox
        
        # Extract board region
        board_region = image[y:y+h, x:x+w]
        
        if board_region.size == 0:
            return None
        
        # Get grayscale versions
        gray_board = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY) if len(board_region.shape) == 3 else board_region
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) if len(reference_img.shape) == 3 else reference_img
        
        # Find the 4 corners of the detected board using feature matching with reference
        source_corners = self._find_board_corners_from_reference(board_region, reference_img)
        
        if source_corners is None or len(source_corners) < 4:
            # Fallback: just return the cropped board with minor rotation correction
            return self._align_rotation(board_region, gray_ref)
        
        # Get reference image dimensions for target
        ref_h, ref_w = reference_img.shape[:2]
        
        # Define target corners (normalized rectangle matching reference dimensions)
        target_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])
        
        try:
            # Get perspective transformation matrix
            M = cv2.getPerspectiveTransform(source_corners, target_corners)
            
            # Apply perspective transformation
            transformed = cv2.warpPerspective(board_region, M, (ref_w, ref_h), 
                                             flags=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_REPLICATE)
            
            return transformed
        except Exception as e:
            # If perspective transformation fails, return rotated version
            return self._align_rotation(board_region, gray_ref)
    
    def _find_board_corners_from_reference(self, board_image: np.ndarray, reference_img: np.ndarray):
        """
        Find the 4 corners of the board by matching reference image corners to the detected board.
        
        Args:
            board_image: Detected board region image
            reference_img: Reference image (board0.png)
            
        Returns:
            Array of 4 corner points in order: top-left, top-right, bottom-right, bottom-left
        """
        gray_board = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY) if len(board_image.shape) == 3 else board_image
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) if len(reference_img.shape) == 3 else reference_img
        
        # Use AKAZE to find features and match
        akaze = cv2.AKAZE_create(nOctaves=4, nOctaveLayers=4)
        
        kp_ref, des_ref = akaze.detectAndCompute(gray_ref, None)
        kp_board, des_board = akaze.detectAndCompute(gray_board, None)
        
        if des_ref is None or des_board is None or len(kp_ref) < 4 or len(kp_board) < 4:
            return None
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_ref, des_board, k=2)
        
        if not matches or len(matches) < 8:
            return None
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 6:
            return None
        
        # Get matched points
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_board[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            return None
        
        if H is None:
            return None
        
        # Get reference image corners
        ref_h, ref_w = gray_ref.shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        
        # Transform reference corners to board image space
        try:
            board_corners = cv2.perspectiveTransform(ref_corners, H)
            board_corners = board_corners.reshape(-1, 2).astype(np.float32)
            
            # Ensure corners are within image bounds and in correct order
            board_h, board_w = board_image.shape[:2]
            
            # Clip to image bounds
            board_corners[:, 0] = np.clip(board_corners[:, 0], 0, board_w - 1)
            board_corners[:, 1] = np.clip(board_corners[:, 1], 0, board_h - 1)
            
            return board_corners
        except:
            return None
    
    def _dewarp_image(self, image: np.ndarray) -> np.ndarray:
        """
        Dewarp image to correct barrel/pincushion distortion and curved edges.
        Uses edge detection to straighten curved board edges.
        
        Args:
            image: Input image
            
        Returns:
            Dewarped image or None if dewarping fails
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Try to detect curves in the edges and straighten them
        # Use morphological operations to enhance edge structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Create a map for distortion correction
        # Use the board edges to calculate distortion parameters
        try:
            # Find horizontal and vertical lines (board edges)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 0:
                # Analyze lines to detect curvature
                # Filter for horizontal and vertical lines
                h_lines = []
                v_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if length > 20:  # Only consider significant lines
                        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                        
                        if angle < 15 or angle > 165:  # Horizontal
                            h_lines.append((y1 + y2) / 2)
                        elif 75 < angle < 105:  # Vertical
                            v_lines.append((x1 + x2) / 2)
                
                # If we detect curved edges, apply correction
                if len(h_lines) > 2 and len(v_lines) > 2:
                    # Check for curvature
                    h_lines = np.array(h_lines)
                    v_lines = np.array(v_lines)
                    
                    # Calculate standard deviation to detect curvature
                    h_std = np.std(np.abs(np.diff(h_lines)))
                    v_std = np.std(np.abs(np.diff(v_lines)))
                    
                    # If significant deviations, apply radial distortion correction
                    if h_std > 5 or v_std > 5:
                        return self._apply_radial_distortion_correction(image)
        except:
            pass
        
        return None
    
    def _apply_radial_distortion_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply radial distortion correction to straighten curved board edges.
        
        Args:
            image: Input image
            
        Returns:
            Corrected image
        """
        h, w = image.shape[:2]
        
        # Create meshgrid for remapping
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Calculate radius from center
        R = np.sqrt(X**2 + Y**2)
        
        # Apply inverse radial correction with mild coefficients
        # This corrects mild barrel distortion common in mobile cameras
        k1 = 0.0005  # Radial distortion coefficient
        
        # Apply correction
        R_corrected = R * (1 + k1 * R**2)
        
        # Normalize back to pixel coordinates
        mask = R != 0
        X_corrected = np.zeros_like(X)
        Y_corrected = np.zeros_like(Y)
        
        X_corrected[mask] = X[mask] * R_corrected[mask] / R[mask]
        Y_corrected[mask] = Y[mask] * R_corrected[mask] / R[mask]
        
        # Convert to pixel coordinates
        map_x = ((X_corrected + 1) * w / 2).astype(np.float32)
        map_y = ((Y_corrected + 1) * h / 2).astype(np.float32)
        
        # Apply remapping
        corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return corrected
    
    def _find_board_corners_precise(self, gray_image: np.ndarray):
        """
        Find the four corners of the board using advanced techniques.
        Uses Harris corner detection combined with edge analysis.
        
        Args:
            gray_image: Grayscale board image
            
        Returns:
            Array of 4 corner points or None if detection fails
        """
        # First try contour-based corner detection
        corners = self._find_board_corners(gray_image)
        
        if corners is not None and len(corners) == 4:
            return corners
        
        # Fallback: Use Harris corner detection
        try:
            # Detect corners using Harris corner detector
            corners_harris = cv2.cornerHarris(gray_image, 2, 3, 0.04)
            corners_harris = cv2.dilate(corners_harris, None)
            
            # Get corner coordinates
            threshold = 0.01 * corners_harris.max()
            corner_coords = np.where(corners_harris > threshold)
            
            if len(corner_coords[0]) < 4:
                return None
            
            # Get the most prominent corners
            corner_points = np.column_stack((corner_coords[1], corner_coords[0]))
            
            # Cluster corners to find 4 main corners
            # Find corners closest to image corners
            h, w = gray_image.shape[:2]
            
            # Define regions (top-left, top-right, bottom-right, bottom-left)
            regions = [
                (0, 0, w//3, h//3),           # Top-left
                (2*w//3, 0, w, h//3),         # Top-right
                (2*w//3, 2*h//3, w, h),       # Bottom-right
                (0, 2*h//3, w, h)             # Bottom-left
            ]
            
            detected_corners = []
            
            for x1, y1, x2, y2 in regions:
                # Find points in this region
                mask = (corner_points[:, 0] >= x1) & (corner_points[:, 0] <= x2) & \
                       (corner_points[:, 1] >= y1) & (corner_points[:, 1] <= y2)
                
                region_corners = corner_points[mask]
                
                if len(region_corners) > 0:
                    # Get the corner with strongest response
                    strengths = corners_harris[region_corners[:, 1].astype(int), region_corners[:, 0].astype(int)]
                    strongest_idx = np.argmax(strengths)
                    detected_corners.append(region_corners[strongest_idx].astype(np.float32))
            
            if len(detected_corners) == 4:
                # Sort corners in order: top-left, top-right, bottom-right, bottom-left
                corners_array = np.array(detected_corners)
                centroid = corners_array.mean(axis=0)
                
                angles = np.arctan2(corners_array[:, 1] - centroid[1], 
                                   corners_array[:, 0] - centroid[0])
                sorted_indices = np.argsort(angles)
                
                return corners_array[sorted_indices]
        except:
            pass
        
        return None
    
    def _find_board_corners(self, gray_image: np.ndarray):
        """
        Find the four corners of the board using edge detection and contours.
        
        Args:
            gray_image: Grayscale board image
            
        Returns:
            Array of 4 corner points or None if detection fails
        """
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour (should be board boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Try to get 4 corners
        if len(approx) >= 4:
            # Sort corners: top-left, top-right, bottom-right, bottom-left
            points = approx.reshape(-1, 2).astype(np.float32)
            
            # Calculate centroid
            centroid = points.mean(axis=0)
            
            # Sort by angle from centroid
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            points = points[sorted_indices]
            
            return points
        
        return None
    
    def _align_rotation(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Align rotation of image to match reference by analyzing edge orientation.
        
        Args:
            image: Image to rotate
            reference: Reference image for angle comparison
            
        Returns:
            Rotated image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use Sobel to detect edges and their angles
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate angles of edges
        angles = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Find dominant angle (most common angle)
        hist, bins = np.histogram(angles.flatten(), bins=180, range=(-90, 90))
        
        # Get angle with highest count
        dominant_angle = bins[np.argmax(hist)]
        
        # Round to nearest 90 degrees (board should be axis-aligned)
        angles_to_try = [dominant_angle, dominant_angle + 90, dominant_angle - 90, dominant_angle + 180]
        
        # Find angle closest to 0 or 90 degrees (axis-aligned)
        best_angle = min(angles_to_try, key=lambda a: min(abs(a % 180), abs((a + 90) % 180)))
        
        # Only rotate if angle is significant (> 3 degrees)
        if abs(best_angle) > 3:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def find_board_region_by_features(self, image: np.ndarray, reference_img: np.ndarray):
        """
        Find the board region using feature matching with reference image.
        Uses AKAZE for scale-invariant robust matching.
        
        Args:
            image: Input image to search
            reference_img: Reference image to match against
            
        Returns:
            (x, y, w, h) bounding box of detected board, or None if not found
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        
        # Initialize AKAZE detector (more robust than ORB, scale-invariant)
        akaze = cv2.AKAZE_create(nOctaves=4, nOctaveLayers=4)
        
        # Find keypoints and descriptors
        kp1, des1 = akaze.detectAndCompute(gray_reference, None)
        kp2, des2 = akaze.detectAndCompute(gray_image, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Fallback to ORB if AKAZE fails
            return self._find_board_with_orb(gray_image, gray_reference, kp1, kp2, des1, des2)
        
        # Create BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors using KNN
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            return None
        
        if not matches or len(matches) < 8:
            return None
        
        # Apply Lowe's ratio test with relaxed threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 6:
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography with relaxed threshold
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
        except:
            return None
        
        if H is None:
            return None
        
        # Get the bounding box of the reference image in the target image
        h, w = gray_reference.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        try:
            dst = cv2.perspectiveTransform(pts, H)
        except:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(dst)
        
        # Ensure coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return None
        
        return (x, y, w, h)
    
    def _find_board_with_orb(self, gray_image, gray_reference, kp1, kp2, des1, des2):
        """Fallback ORB matching when AKAZE fails"""
        # Initialize ORB detector with many features
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray_reference, None)
        kp2, des2 = orb.detectAndCompute(gray_image, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        
        # Create BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            return None
        
        if not matches or len(matches) < 8:
            return None
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 6:
            return None
        
        # Extract matched points and find homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
        except:
            return None
        
        if H is None:
            return None
        
        # Get bounding box
        h, w = gray_reference.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        try:
            dst = cv2.perspectiveTransform(pts, H)
        except:
            return None
        
        x, y, w, h = cv2.boundingRect(dst)
        
        # Ensure coordinates are within bounds
        img_h, img_w = gray_image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return None
        
        return (x, y, w, h)
    
    def find_board_region_by_shape_matching(self, image: np.ndarray, reference_img: np.ndarray):
        """
        Find the board region using shape matching (rotation-invariant).
        Detects the board based on its contour shape, handles rotation.
        
        Args:
            image: Input image to search
            reference_img: Reference image for shape comparison
            
        Returns:
            (x, y, w, h) bounding box of detected board, or None if not found
        """
        # Get reference board contour
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        _, binary_ref = cv2.threshold(gray_ref, 100, 255, cv2.THRESH_BINARY)
        binary_ref = cv2.bitwise_not(binary_ref)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_ref = cv2.morphologyEx(binary_ref, cv2.MORPH_CLOSE, kernel)
        
        contours_ref, _ = cv2.findContours(binary_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_ref:
            return None
        
        ref_contour = max(contours_ref, key=cv2.contourArea)
        ref_area = cv2.contourArea(ref_contour)
        
        # Find contours in target image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
        binary_img = cv2.bitwise_not(binary_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        
        contours_img, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_img:
            return None
        
        # Find contour with shape most similar to reference
        best_contour = None
        best_distance = float('inf')
        
        for contour in contours_img:
            area = cv2.contourArea(contour)
            # Only consider contours with similar area (within 30-500% of reference)
            if area < ref_area * 0.3 or area > ref_area * 5.0:
                continue
            
            # Compare shape
            distance = cv2.matchShapes(ref_contour, contour, cv2.CONTOURS_MATCH_I3, 0)
            
            if distance < best_distance:
                best_distance = distance
                best_contour = contour
        
        # Accept if match distance is reasonable (lower is better)
        if best_contour is None or best_distance > 0.3:
            return None
        
        x, y, w, h = cv2.boundingRect(best_contour)
        return (x, y, w, h)
        """
        Find the board region by detecting contours using multiple edge detection methods.
        Used as fallback when feature matching fails.
        
        Args:
            image: Input image
            
        Returns:
            (x, y, w, h) bounding box of board, or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection approaches and pick the best result
        candidates = []
        
        # Method 1: Threshold-based detection
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours1:
            largest = max(contours1, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            area = w * h
            img_area = image.shape[0] * image.shape[1]
            if area > (img_area * 0.05):  # At least 5% of image
                candidates.append(((x, y, w, h), area))
        
        # Method 2: Canny edge detection with adaptive thresholds
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        for canny_low, canny_high in [(20, 60), (30, 100), (50, 150)]:
            edges = cv2.Canny(filtered, canny_low, canny_high)
            kernel_size = max(5, int(min(image.shape) * 0.02))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours2:
                largest = max(contours2, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                area = w * h
                img_area = image.shape[0] * image.shape[1]
                if area > (img_area * 0.05):  # At least 5% of image
                    candidates.append(((x, y, w, h), area))
        
        if not candidates:
            return None
        
        # Pick the largest candidate (most likely to be the board)
        best_bbox, _ = max(candidates, key=lambda x: x[1])
        
        return best_bbox
    
    def crop_image(self, image: np.ndarray, bbox):
        """
        Crop image to bounding box.
        
        Args:
            image: Input image
            bbox: (x, y, w, h) bounding box
            
        Returns:
            Cropped image
        """
        x, y, w, h = bbox
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
        
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def get_single_board_images(self):
        """
        Get all board*.png files from source directory, excluding reference image.
        
        Returns:
            List of Path objects for board*.png files
        """
        image_files = sorted([
            f for f in self.source_dir.glob("board*.png")
            if f.name != self.reference_image
        ])
        return image_files
    
    def clean_image(self, image_path: Path):
        """
        Clean a single image by detecting board using multiple methods:
        1. Feature matching (handles rotation and scale)
        2. Shape matching (rotation-invariant)
        3. Edge detection (fallback)
        Then uses board0 reference to find corners and apply perspective transformation.
        
        Args:
            image_path: Path to image
            
        Returns:
            True if successful, False otherwise
        """
        print(f"▶ {image_path.name}", end=" ")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("✗ Could not load")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{image_path.name}: Could not load")
            return False
        
        original_shape = image.shape
        
        # Try feature matching first (handles rotation and scale)
        bbox = self.find_board_region_by_features(image, self.reference_img)
        
        # Fallback to shape matching if feature matching fails (rotation-invariant)
        if bbox is None:
            bbox = self.find_board_region_by_shape_matching(image, self.reference_img)
        
        # Final fallback to edge detection
        if bbox is None:
            bbox = self.find_board_region_by_edges(image)
        
        if bbox is None:
            print("✗ Could not detect board")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{image_path.name}: Could not detect board")
            return False
        
        # Crop image to board region
        cropped = self.crop_image(image, bbox)
        
        if cropped is None:
            print("✗ Could not crop image")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{image_path.name}: Could not crop")
            return False
        
        # Apply perspective correction and rotation alignment
        corrected = self.correct_perspective_and_rotation(image, self.reference_img, bbox)
        if corrected is not None:
            cropped = corrected
        
        new_shape = cropped.shape
        
        # Save cleaned image
        output_path = self.output_dir / image_path.name
        cv2.imwrite(str(output_path), cropped)
        
        print(f"✓ {original_shape[1]}×{original_shape[0]} → {new_shape[1]}×{new_shape[0]}")
        
        self.stats['matched'] += 1
        return True
    
    def run(self):
        """Process all board*.png images"""
        print("\n" + "="*80)
        print("CLEANING SINGLE BOARD IMAGES FROM SOURCE")
        print("="*80)
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Reference image: {self.reference_image}\n")
        
        # Copy reference image first
        self.copy_reference_image()
        
        # Get single board images
        board_images = self.get_single_board_images()
        
        if not board_images:
            print("\n✗ No other board*.png files found")
            return False
        
        print(f"\nFound {len(board_images)} image(s) to clean:\n")
        for img in board_images:
            print(f"  • {img.name}")
        
        print("\nProcessing...\n")
        
        # Process each image
        for img_path in board_images:
            self.clean_image(img_path)
        
        # Print summary
        print("\n" + "="*80)
        print("CLEANING SUMMARY")
        print("="*80)
        total = len(board_images) + self.stats['copied']
        print(f"Copied (reference): {self.stats['copied']}")
        print(f"Matched & cropped: {self.stats['matched']}/{len(board_images)}")
        print(f"Failed: {self.stats['failed']}/{len(board_images)}")
        
        if self.stats['errors']:
            print("\nErrors:")
            for error in self.stats['errors']:
                print(f"  ✗ {error}")
        
        print(f"\n✓ Cleaned images saved to: {self.output_dir}/")
        
        return self.stats['failed'] == 0


def main():
    """Main entry point"""
    # Parse command line arguments
    source_dir = "training_data"
    output_dir = None
    reference_image = "board0.png"
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--source' and i + 1 < len(sys.argv) - 1:
            source_dir = sys.argv[i + 2]
        elif arg == '--output' and i + 1 < len(sys.argv) - 1:
            output_dir = sys.argv[i + 2]
        elif arg == '--reference' and i + 1 < len(sys.argv) - 1:
            reference_image = sys.argv[i + 2]
        elif arg in ['-h', '--help']:
            print("Clean Single Board Images Module - Board0 Reference-Based Perspective Transform")
            print("\nUsage:")
            print("  python clean_source_boards.py")
            print("  python clean_source_boards.py --output training_data/cleaned")
            print("  python clean_source_boards.py --reference board0.png")
            print("\nOptions:")
            print("  --source DIR       Source directory (default: training_data)")
            print("  --output DIR       Output directory (default: training_output/cleaned)")
            print("  --reference FILE   Reference image name (default: board0.png)")
            print("  -h, --help         Show this help message")
            print("\nDetection Methods (in order):")
            print("  1. Feature matching (AKAZE/ORB) - handles rotation and scale")
            print("  2. Shape matching (contours) - rotation-invariant")
            print("  3. Edge detection - fallback for difficult cases")
            print("\nNormalization Method:")
            print("  1. Match features between board0 and detected board")
            print("  2. Find homography to map board0 corners to detected board")
            print("  3. Apply perspective transformation to normalize orientation")
            print("\nProcess:")
            print("  1. Copies reference image to output (ideal image)")
            print("  2. Detects board using multi-method approach")
            print("  3. Finds 4 corners using board0 as reference")
            print("  4. Applies OpenCV perspective transformation")
            print("  5. Crops and saves normalized board region")
            sys.exit(0)
    
    try:
        # Run cleaner
        cleaner = SourceBoardCleaner(source_dir, output_dir, reference_image)
        success = cleaner.run()
        sys.exit(0 if success else 1)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
