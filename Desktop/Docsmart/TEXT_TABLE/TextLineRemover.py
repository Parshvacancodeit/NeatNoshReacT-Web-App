import cv2
import numpy as np
import os

class TextLineRemover:
    def __init__(self, image, save_debug_images=True):
        """
        Initialize the TableLinesRemover with the given image.
        :param image: The input image to process.
        :param save_debug_images: Whether to save intermediate processing images.
        """
        self.image = image
        self.save_debug_images = save_debug_images

    def execute(self):
        """
        Execute the line removal process.
        :return: Image with table lines removed, ready for OCR.
        """
        self.grayscale_image()
        self.store_process_image("0_grayscaled.jpg", self.grey)

        self.blurred_image()
        self.store_process_image("11_blurred.jpg", self.blurred)

        self.threshold_image()
        self.store_process_image("1_thresholded.jpg", self.thresholded_image)

        self.invert_image()
        self.store_process_image("2_inverted.jpg", self.inverted_image)

        self.erode_vertical_lines()
        self.store_process_image("3_eroded_vertical_lines.jpg", self.vertical_lines_eroded_image)

        self.erode_horizontal_lines()
        self.store_process_image("4_eroded_horizontal_lines.jpg", self.horizontal_lines_eroded_image)

        self.combine_eroded_images()
        self.store_process_image("5_combined_eroded_images.jpg", self.combined_image)

        self.dilate_combined_image_to_make_lines_thicker()
        self.store_process_image("6_dilated_combined_image.jpg", self.combined_image_dilated)

        self.subtract_combined_and_dilated_image_from_original_image()
        self.store_process_image("7_image_without_lines.jpg", self.image_without_lines)

        self.remove_noise_with_erode_and_dilate()
        self.store_process_image("8_image_without_lines_noise_removed.jpg", self.image_without_lines_noise_removed)

        return self.image_without_lines_noise_removed

    def grayscale_image(self):
        """Convert the image to grayscale."""
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def blurred_image(self):

         self.blurred = cv2.GaussianBlur(self.grey, (7, 7), 11)

        

    def threshold_image(self):
        """Apply binary thresholding to the grayscale image."""
        self.thresholded_image = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,4
        )

    def invert_image(self):
        """Invert the binary thresholded image."""
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def erode_vertical_lines(self):
        """Erode vertical lines in the image to isolate them."""
        vertical_kernel = np.array([[1],[1],[1],[1], [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, vertical_kernel, iterations=4)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, vertical_kernel, iterations=35)

    def erode_horizontal_lines(self):
        """Erode horizontal lines in the image to isolate them."""
        horizontal_kernel = np.array([[1,1,1,1,1,1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, horizontal_kernel, iterations=7)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, horizontal_kernel, iterations=22)

    def combine_eroded_images(self):
        """Combine the eroded vertical and horizontal lines into one image."""
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

    def dilate_combined_image_to_make_lines_thicker(self):
        """Dilate the combined lines to make them thicker for better subtraction."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=2)

    def subtract_combined_and_dilated_image_from_original_image(self):
        """Subtract the combined, dilated lines from the inverted image."""
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)

    def remove_noise_with_erode_and_dilate(self):
        """Remove noise from the image using erode and dilate operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=2)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=2)

    def store_process_image(self, file_name, image):
        """
        Save the processed image to the specified file.
        """
        if not self.save_debug_images:
            return

        # Define the directory
        dir_path = "./Bholenath/table_lines_remover_bh/"
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Construct the file path
        path = os.path.join(dir_path, file_name)
        
        # Save the image
        success = cv2.imwrite(path, image)
        if not success:
            print(f"Failed to save image: {path}")
        else:
            print(f"Image saved successfully: {path}")


