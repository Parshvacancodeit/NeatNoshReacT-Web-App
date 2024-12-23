import cv2
import numpy as np

class SignatureRemover:
    def __init__(self, image):
        """
        Initialize with the image to process.
        :param image: The input image to process.
        """
        self.image = image
        self.processed_image = image.copy()
        self.rows=[]

    def remove_blue_signature(self):
        """
        Detect and subtract blue signature from the image while preserving text.
        :return: Image with blue signature removed (preserved text).
        """
        # Convert image to HSV (Hue, Saturation, Value) for easier color filtering
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Debug: Show the HSV image to understand the range of blue colors
        cv2.imshow("HSV Image", hsv_image)

        # Define the range of blue in HSV space
        lower_blue = np.array([100, 150, 50])  # Lower bound of blue
        upper_blue = np.array([140, 255, 255])  # Upper bound of blue

        # Create a mask for the blue areas
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Debug: Show the blue mask to visualize the detected blue areas
        cv2.imshow("Blue Mask", blue_mask)

        # Perform dilation to make sure we cover the signature areas
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(blue_mask, kernel, iterations=2)

        # Debug: Show the dilated blue mask
        cv2.imshow("Dilated Blue Mask", dilated_mask)

        # Subtract the dilated blue signature areas from the original image
        # This is similar to your line-removal technique
        self.processed_image = cv2.subtract(self.image, cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR))

        # Debug: Show the final image with the blue signature removed
        cv2.imshow("Image Without Blue Signature", self.processed_image)

        return self.processed_image
