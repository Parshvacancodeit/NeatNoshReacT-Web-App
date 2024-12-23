# #TableExtractor.py

# import cv2
# import numpy as np

# class TableExtractor:
#     def __init__(self, path_to_image):
#         self.path_to_image = path_to_image
#         self.image         = cv2.imread(path_to_image)
#         if self.image is None:
#             raise FileNotFoundError(f"Image not found at path: {path_to_image}")

#     def execute(self):
#         self.store_process_image("0_original.jpg", self.image)
#         self.find_contours()
#         self.order_points_in_the_contour_with_max_area()
#         self.warp_perspective()
#         self.store_process_image("1_perspective_corrected.jpg", self.perspective_corrected_image)
#         return self.perspective_corrected_image

#     def find_contours(self):
#         """Find contours in the processed image."""
#         # Convert the image to grayscale
#         grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
#         # Apply GaussianBlur to reduce noise and enhance contour detection
#         blurred = cv2.GaussianBlur(grey, (5, 5), 0)

#         # Apply adaptive thresholding for better handling of varying lighting conditions
#         thresholded = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#         )

#         # Find contours
#         contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # Debugging: Print the number of contours found
#         print(f"Number of contours found: {len(contours)}")

#         if len(contours) == 0:
#             raise ValueError("No contours found in the image.")

#         # Save debugging images if necessary
#         if hasattr(self, 'store_process_image'):
#             self.store_process_image("thresholded.jpg", thresholded)
#             contour_image = self.image.copy()
#             cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
#             self.store_process_image("contours.jpg", contour_image)

#         # Assign contours to the instance variable for further use
#         self.contours = contours


#     def order_points_in_the_contour_with_max_area(self):
#         """Order points of the largest contour by area."""
#         # Find the contour with the maximum area
#         self.contour_with_max_area = max(self.contours, key=cv2.contourArea)

#         # Ensure the contour has enough points
#         if len(self.contour_with_max_area) < 4:
#             raise ValueError("Contour with max area does not have enough points for ordering.")

#         # Reshape the contour to (N, 2)
#         self.contour_with_max_area = self.contour_with_max_area.reshape(-1, 2)
#         print(f"Max contour points shape: {self.contour_with_max_area.shape}")

#         # Order points
#         self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)

#     def order_points(self, pts):
#         """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
#         if len(pts) < 4:
#             raise ValueError("Not enough points to perform ordering. At least 4 points are required.")

#         rect = np.zeros((4, 2), dtype="float32")

#         # Compute sum and difference of points
#         s = pts.sum(axis=1)
#         diff = np.diff(pts, axis=1)

#         if len(s) < 4 or len(diff) < 4:
#             raise ValueError(f"Invalid points for ordering. Sum: {s}, Diff: {diff}")

#         rect[0] = pts[np.argmin(s)]  # Top-left
#         rect[2] = pts[np.argmax(s)]  # Bottom-right
#         rect[1] = pts[np.argmin(diff)]  # Top-right
#         rect[3] = pts[np.argmax(diff)]  # Bottom-left

#         return rect

#     def warp_perspective(self):
#         """Warp the perspective of the image using the ordered points."""
#         rect = self.contour_with_max_area_ordered
#         (tl, tr, br, bl) = rect

#         # Compute width and height
#         width_a = np.linalg.norm(br - bl)
#         width_b = np.linalg.norm(tr - tl)
#         max_width = max(int(width_a), int(width_b))

#         height_a = np.linalg.norm(tr - br)
#         height_b = np.linalg.norm(tl - bl)
#         max_height = max(int(height_a), int(height_b))

#         # Destination points
#         dst = np.array([
#             [0, 0],
#             [max_width - 1, 0],
#             [max_width - 1, max_height - 1],
#             [0, max_height - 1]
#         ], dtype="float32")

#         # Perspective transform
#         matrix = cv2.getPerspectiveTransform(rect, dst)
#         self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (max_width, max_height))

#     def store_process_image(self, file_name, image):
#         """Save intermediate processing images for debugging."""
#         path    = "./process_images/table_extractor/" + file_name
#         success = cv2.imwrite(path, image)
#         if not success:
#             raise IOError(f"Failed to write image to path: {path}")

import os 
import cv2
import numpy as np

class TableExtractor:
    def __init__(self, path_to_image, save_debug_images=True):
        """
        Initialize the TableExtractor.
        :param path_to_image: Path to the input image.
        :param save_debug_images: Whether to save intermediate processing images.
        """
        self.path_to_image = path_to_image
        self.image = cv2.imread(path_to_image)
        self.save_debug_images = save_debug_images

        if self.image is None:
            raise FileNotFoundError(f"Image not found at path: {path_to_image}")

    def execute(self):
        """
        Main function to execute the table extraction process.
        :return: Perspective-corrected image of the table.
        """
        if self.save_debug_images:
            self.store_process_image("0_original.jpg", self.image)

        self.find_contours()
        self.order_points_in_the_contour_with_max_area()
        self.warp_perspective()

        if self.save_debug_images:
            self.store_process_image("1_perspective_corrected.jpg", self.perspective_corrected_image)

        return self.perspective_corrected_image

    def find_contours(self):
        """Find contours in the processed image."""
        # Convert the image to grayscale
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and enhance contour detection
        blurred = cv2.GaussianBlur(grey, (5, 5), 4)

        # Apply adaptive thresholding for better handling of varying lighting conditions
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,5
        )

        dilated_image = cv2.dilate(thresholded, None, iterations=7)

        # Find contours
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Debugging: Print the number of contours found
        print(f"Number of contours found: {len(contours)}")

        if len(contours) == 0:
            raise ValueError("No contours found in the image.")

        # Save debugging images if necessary
        if self.save_debug_images:
            self.store_process_image("thresholded.jpg", thresholded)
            contour_image = self.image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            self.store_process_image("contours.jpg", contour_image)

        # Assign contours to the instance variable for further use
        self.contours = contours

    def order_points_in_the_contour_with_max_area(self):
        """Order points of the largest contour by area."""
        # Find the contour with the maximum area
        self.contour_with_max_area = max(self.contours, key=cv2.contourArea)

        # Ensure the contour has enough points
        if len(self.contour_with_max_area) < 4:
            raise ValueError("Contour with max area does not have enough points for ordering.")

        # Reshape the contour to (N, 2)
        self.contour_with_max_area = self.contour_with_max_area.reshape(-1, 2)
        print(f"Max contour points shape: {self.contour_with_max_area.shape}")

        # Order points
        self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)

    def order_points(self, pts):
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        if len(pts) < 4:
            raise ValueError("Not enough points to perform ordering. At least 4 points are required.")

        rect = np.zeros((4, 2), dtype="float32")

        # Compute sum and difference of points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        if len(s) < 4 or len(diff) < 4:
            raise ValueError(f"Invalid points for ordering. Sum: {s}, Diff: {diff}")

        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def warp_perspective(self):
        """Warp the perspective of the image using the ordered points."""
        rect = self.contour_with_max_area_ordered
        (tl, tr, br, bl) = rect

        # Compute width and height
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        # Destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(rect, dst)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (max_width, max_height))

    def store_process_image(self, file_name, image):
        """Save intermediate processing images for debugging."""
        if not self.save_debug_images:
            return

        path = "./Bholenath/table_extractor_bh/" + file_name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        success = cv2.imwrite(path, image)
        if not success:
            raise IOError(f"Failed to write image to path: {path}")
