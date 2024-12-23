import numpy as np
import cv2

class BarcodeRemover:
    def __init__(self, image):
        """
        Initializes the BarcodeRemover with the image.

        :param image: Input image as a numpy array
        """
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def preprocess_image(self):
        """
        Preprocesses the image to prepare for barcode extraction.

        :return: Thresholded and morphologically processed image
        """
        # Apply binary thresholding
        (_, thresh) = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY_INV)

        width = 20

        # Construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, width // 2 + 1))
        E = cv2.erode(thresh, kernel, iterations=1)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, width // 2 + 1))
        D = cv2.dilate(E, kernel2, iterations=2)

        return D

    def remove_barcodes(self):
        """
        Detects the largest barcode in the image and replaces it with white color.

        :return: Modified image with the largest barcode replaced by white color, or original image if no barcode is found.
        """
        # Preprocess the image
        D = self.preprocess_image()

        # Find contours in the thresholded image
        (cnts, _) = cv2.findContours(D.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if there are any contours detected
        if not cnts:
            print("No contours detected")
            return self.image

        # Sort contours by area and get the bounding rectangle of the largest one
        largest_contour = max(cnts, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the area of the largest contour
        area = w * h

        # Check if the contour is large enough and roughly square-shaped
        if area < 35000 or abs(w - h) > min(w, h) * 0.2:
            print("Detected contour is either too small or not square enough")
            return self.image

        # Print the area of the largest barcode
        print(f"Area of the largest detected barcode: {area} pixels")

        # Replace the largest barcode area with white color
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)  # Fill with white

        return self.image

    # def handle_vertical_barcodes(self):
    #     """
    #     Detects vertical barcodes and replaces them with white color without rotating or cropping the image.
    #     Also draws green boxes around detected vertical barcodes.

    #     :return: Modified image with vertical barcodes replaced by white color
    #     """
    #     # Preprocess the image to get the thresholded image
    #     D = self.preprocess_image()

    #     # Find contours in the thresholded image
    #     (cnts, _) = cv2.findContours(D.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     for index in range(len(cnts)):
    #         x, y, w1, h1 = cv2.boundingRect(cnts[index])

    #         # Only consider vertical contours (barcode candidates) based on aspect ratio
    #         if cv2.contourArea(cnts[index]) > 100 and w1 > h1:  # Only vertical barcodes with enough area
    #             # Draw a green rectangle around the detected vertical barcode
    #             cv2.rectangle(self.image, (x, y), (x + w1, y + h1), (0, 255, 0), 2)  # Green color, thickness 2

    #     return self.image