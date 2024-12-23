import cv2
import numpy as np
import pytesseract
import re
import os
import json
import subprocess

class CropTextRegion:
    def __init__(self, image, original_image,table_start_y=None):
        if image is None or original_image is None:
            raise ValueError("Input image or original image is None. Please check the provided images.")
        self.image = image
        self.original_image = original_image
        self.table_start_y = table_start_y
        
        # Crop from top to the starting dimension of the table
        cropped_image = self.image[:self.table_start_y, :]
        
        
        
        # Update self.image to the cropped image
        self.image = cropped_image
    def execute(self):
        return self.image