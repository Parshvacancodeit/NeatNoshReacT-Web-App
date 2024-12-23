import os
import cv2
import numpy as np

class DetectTable:
    def __init__(self, path_to_image, save_debug_images=True):
        """
        Initialize the DetectTable.
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
        Main function to execute the table detection process.
        :return: Cropped table image that is passed to the line remover.
        """
        if self.save_debug_images:
            self.store_process_image("0_original.jpg", self.image)

        # Detect table using enhanced preprocessing
        detected_table_image = self.detect_table_cells()

        if self.save_debug_images:
            self.store_process_image("detected_table.jpg", detected_table_image)

        # Crop the detected table
        cropped_table_image = self.crop_table(detected_table_image)

        # Return the cropped image for further processing (like line removal)
        return cropped_table_image

    def detect_table_cells(self):
        """
        Detect the table using improved preprocessing.
        :return: Image with detected table.
        """
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 2)
        # Binarize the image using adaptive thresholding and global thresholding
        self.thresholded_image = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,3
        )

        # Apply Canny edge detection to detect edges more clearly
        edges = cv2.Canny(self.thresholded_image, 50, 150)

        if self.save_debug_images:
            self.store_process_image("edges.jpg", edges)

        # Use morphological operations to enhance the table's edges
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel to close bigger gaps
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if self.save_debug_images:
            self.store_process_image("morph.jpg", morph)

        # Find contours in the processed image
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debugging: Show how many contours are found
        print(f"Number of contours found: {len(contours)}")

        # Filter contours to find the table (based on area, aspect ratio, and shape)
        table_contour = self.get_table_contour(contours)

        # Crop the image based on the table contour
        if table_contour is not None:
            x, y, w, h = cv2.boundingRect(table_contour)
            cropped_image = self.image[y:y+h, x:x+w]
            return cropped_image
        else:
            raise ValueError("No table found in the image.")

    def get_table_contour(self, contours):
        """
        Filter contours and find the one corresponding to the table.
        :param contours: List of contours found in the image.
        :return: The contour of the table if found, otherwise None.
        """
        for contour in contours:
            # Filter based on contour area (ignore small contours)
            area = cv2.contourArea(contour)
            if area > 5000:  # Increase threshold if necessary
                # Optionally, filter by aspect ratio (e.g., table-like shape)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                # Check if the contour is approximately rectangular (a table-like shape)
                if len(approx) == 4:  # Approximate a polygon with 4 vertices (rectangle)
                    return contour
        return None

    def crop_table(self, detected_table_image):
        """
        Crop the image based on detected table position.
        :param detected_table_image: Image with detected table
        :return: Cropped table image
        """
        return detected_table_image

    def store_process_image(self, file_name, image):
        """Save intermediate processing images for debugging."""
        if not self.save_debug_images:
            return

        path = "./Bholenath/detect_table_debug/" + file_name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        success = cv2.imwrite(path, image)
        if not success:
            raise IOError(f"Failed to write image to path: {path}")
