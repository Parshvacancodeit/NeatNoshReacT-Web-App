#<<<----------------IMPORTANT CODE --------------------------->>>



# import cv2
# import numpy as np
# import pytesseract
# import re
# import os
# import json
# import subprocess

# class TextExtractor:
#     def __init__(self, image, original_image, table_start_y=None):
        
#         if image is None or original_image is None:
#             raise ValueError("Input image or original image is None. Please check the provided images.")
        
#         self.image = image
#         self.original_image = original_image
#         self.table_start_y = table_start_y
        
#         # Crop from top to the starting dimension of the table
#         cropped_image = self.image[:self.table_start_y, :]
        
#         # Show the cropped image for verification
#         cv2.imshow("Cropped Image", cropped_image)
#         cv2.waitKey(0)
        
#         # Update self.image to the cropped image
#         self.image = cropped_image

#     def execute(self):
#         # Preprocess the image
#         self.preprocess_image()

        
#         # Detect and label rectangles
#         self.detect_rectangles()
        
#         # Store the processed (cropped and rectangled) image
#         self.store_processed_image("00LM_preprocessed.jpg", self.image)

#     def preprocess_image(self):
#         """Convert image to grayscale and apply adaptive thresholding."""
#         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
#         self.image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
#         cv2.imshow("imggg",self.image)

#     def detect_rectangles(self):
#         """Detect and label rectangles in the image."""
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # Apply Canny edge detection
#         edges = cv2.Canny(gray, 50, 150)

#         # Find contours
#         contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # Iterate over each contour
#         rectangle_id = 1
#         for contour in contours:
#             # Approximate the contour
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)

#             # Filter for rectangles (4 points)
#             if len(approx) == 4:
#                 # Get bounding box coordinates
#                 x, y, w, h = cv2.boundingRect(approx)

#                 # Draw the rectangle on the image
#                 cv2.drawContours(self.image, [approx], -1, (0, 255, 0), 2)

#                 # Label the rectangle with its ID
#                 cv2.putText(self.image, f"ID: {rectangle_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
#                 # Increment rectangle ID
#                 rectangle_id += 1

#         # Display the result
#         cv2.imshow("Detected Rectangles", self.image)
#         cv2.waitKey(0)

#     def store_processed_image(self, filename, image):
#         """Save the processed image."""
#         cv2.imwrite(filename, image)


#         #Perform dilation and contour detection
#         self.dilate_image()
#         self.store_processed_image("011LM_dilated.jpg", self.dilated_image)

#         self.find_contours()
#         self.store_processed_image("022LM_contours.jpg", self.image_with_contours_drawn)

#         self.convert_contours_to_bounding_boxes()
#         self.store_processed_image("033LM_bounding_boxes.jpg", self.image_with_all_bounding_boxes)

#         # Process bounding boxes
#         self.mean_height = self.get_mean_height_of_bounding_boxes()
#         self.sort_bounding_boxes_by_y_coordinate()
#         self.group_bounding_boxes_into_rows()
#         self.sort_rows_by_x_coordinate()

#         # Crop bounding boxes, perform OCR, and exclude table data
#         self.crop_bounding_boxes_and_perform_ocr()
#         self.generate_csv_file()
#         self.generate_json_response()
#         self.print_table_to_console()

#         # Extract text using OCR
#         text_data    = self.extract_text()
#         clean_data   = self.postprocess_text(text_data)
#         return clean_data

#     def preprocess_image(self):
#         """Convert image to grayscale and apply adaptive thresholding."""
#         gray       = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
#         self.image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

#     def dilate_image(self):
#         """Apply dilation to emphasize text regions."""
#         kernel             = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#         self.dilated_image = cv2.dilate(self.image, kernel, iterations=2)

#     def find_contours(self):
#         """Detect contours and draw them on a copy of the original image."""
#         contours, _   = cv2.findContours(self.dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         self.contours = contours
#         self.image_with_contours_drawn = self.original_image.copy()
#         cv2.drawContours(self.image_with_contours_drawn, contours, -1, (0, 255, 0), 2)

#     def convert_contours_to_bounding_boxes(self):
#         """Convert contours into bounding boxes and draw them on the image."""
#         self.bounding_boxes = []
#         self.image_with_all_bounding_boxes = self.original_image.copy()

#         for contour in self.contours:
#             x, y, w, h   = cv2.boundingRect(contour)
#             aspect_ratio = w / float(h)

#             if h > 10 and w > 10 and 0.2 < aspect_ratio < 10:  # Filter non-text regions
#                 self.bounding_boxes.append((x, y, w, h))
#                 cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     def get_mean_height_of_bounding_boxes(self):
#         """Calculate the average height of all bounding boxes."""
#         return np.mean([h for _, _, _, h in self.bounding_boxes])

#     def sort_bounding_boxes_by_y_coordinate(self):
#         """Sort bounding boxes by their vertical position (y-coordinate)."""
#         self.bounding_boxes.sort(key=lambda box: box[1])

#     def group_bounding_boxes_into_rows(self):
#         """Group bounding boxes into rows based on their y-coordinates."""
#         self.rows   = []
#         current_row = [self.bounding_boxes[0]]
#         threshold   = self.mean_height / 2

#         for box in self.bounding_boxes[1:]:
#             if abs(box[1] - current_row[-1][1]) <= threshold:
#                 current_row.append(box)
#             else:
#                 self.rows.append(current_row)
#                 current_row = [box]

#         self.rows.append(current_row)

#     def sort_rows_by_x_coordinate(self):
#         """Sort bounding boxes within each row by their x-coordinates."""
#         for row in self.rows:
#             row.sort(key=lambda box: box[0])

#     def crop_bounding_boxes_and_perform_ocr(self):
#         """Crop each bounding box, perform OCR, and filter table regions."""
#         self.table          = []
#         image_number        = 0
#         table_start_markers = ["Sr.No", "SI.No", "Sr.", "Sr NO"]  # Define table start markers
#         table_detected      = False

#         for row in self.rows:
#             current_row = []
#             for x, y, w, h in row:
#                 cropped_image    = self.original_image[y:y + h, x:x + w]
#                 image_slice_path = f"./ocr_slices/img_{image_number}.jpg"

#                 os.makedirs("./ocr_slices", exist_ok=True)
#                 cv2.imwrite(image_slice_path, cropped_image)

#                 ocr_result = self.get_result_from_tesseract(image_slice_path)

#                 # Check if the OCR result indicates the start of a table
#                 if any(marker.lower() in ocr_result.lower() for marker in table_start_markers):
#                     table_detected = True
#                     break

#                 if not table_detected:
#                     current_row.append(ocr_result)

#                 image_number += 1

#             if not table_detected and current_row:
#                 self.table.append(current_row)

#     def get_result_from_tesseract(self, image_path):
#         """Run Tesseract OCR on an image and return the result."""
#         cmd    = f'tesseract {image_path} - -l eng --oem 3 --psm 6'
#         output = subprocess.getoutput(cmd)
#         return output.strip()

#     def generate_csv_file(self):
#         """Save OCR results to a CSV file."""
#         with open("output_LM11.csv", "w") as file:
#             for row in self.table:
#                 file.write(",".join(row) + "\n")

#     def generate_json_response(self):
#         """Save OCR results to a JSON file."""
#         with open("output_LM11.json", "w") as json_file:
#             json.dump(self.table, json_file, indent=4)

#     def print_table_to_console(self):
#         """Print OCR results in a table format."""
#         print("\nExtracted Table Data:")
#         for row in self.table:
#             print(", ".join(row))

#     def store_processed_image(self, file_name, image):
#         """Save intermediate processing images for debugging."""
#         path = f"./AksharIspat/text_extractor/{file_name}"
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         cv2.imwrite(path, image)

#     def extract_text(self):
#         """Extract text from the entire preprocessed image using Tesseract."""
#         custom_config = r'--oem 3 --psm 6'
#         raw_text      = pytesseract.image_to_string(self.image, config=custom_config)

#         # Exclude text related to the table section
#         table_start_markers = ["Sr.No", "SI.No", "Sr.", "Sr NO"]
#         filtered_lines      = []
#         table_detected      = False

#         for line in raw_text.split("\n"):
#             if any(marker.lower() in line.lower() for marker in table_start_markers):
#                 table_detected = True
#                 break
#             filtered_lines.append(line)

#         return "\n".join(filtered_lines)

#     def postprocess_text(self, text_data):
#         """Clean up the extracted text data."""
#         lines         = text_data.split("\n")
#         cleaned_lines = [line.strip() for line in lines if line.strip()]
#         return "\n".join(cleaned_lines) 

# import re
# #GEMINI
# def extract_key_data_from_text(extracted_text):
#     """Extract and structure key data, handling multiple contact numbers per line."""
#     structured_data = {
#         "Party Name"     : None,
#         "Party Address"  : None,
#         "Contact Numbers": [],
#         "GSTIN": None,
#         "PAN"  : None,
#         "Invoice No"  : None,
#         "Invoice Date": None,
#         "State Code"  : None,
#         "State"          : None,
#         "Place of Supply": None,
#         "Buyer Name"   : None,
#         "Buyer Address": None,
#         "Buyer GSTIN"  : None
#     }

#     patterns = {
#         "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
#         "PAN"  : r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",
#         "Phone": r"\b\d{10}\b",
#         "Invoice No"  : r"Invoice No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
#         "Invoice Date": r"Invoice Date\:\s*(\d{2}/\d{2}/\d{4})",
#         "State Code"  : r"State Code\s*[:\-]?\s*(\d+)",
#         "State"          : r"State\s*[:\-]?\s*([A-Za-z\s]+)",
#         "Place of Supply": r"Place of Supply\s*[:\-]?\s*(\d+)\s*\-\s*([A-Za-z\s]+)"
#     }

#     supplier_keywords = ["to", "m/s", "mr.", "mrs.", "M/s."]
#     lines             = extracted_text.split("\n")

#     party_name_detected    = False
#     address_lines          = []
#     supplier_name_detected = False
#     supplier_address_lines = []
#     supplier_gstin_found   = False

#     for line in lines:
#         clean_line = line.strip()
#         if not clean_line:
#             continue

#         # Extract multiple phone numbers from a single line
#         phone_matches = re.findall(patterns["Phone"], clean_line)
#         structured_data["Contact Numbers"].extend(phone_matches)  # Use extend to add multiple

#         for key, pattern in patterns.items():
#             if key == "Phone": #skip phone because it is already handled above
#                 continue

#             match = re.search(pattern, clean_line)
#             if match:
#                 if key == "Place of Supply":
#                     structured_data[key] = {"Code": match.group(1), "State": match.group(2).strip()}
#                 elif key == "GSTIN":
#                     if supplier_name_detected and not supplier_gstin_found:
#                         structured_data["Supplier GSTIN"] = match.group(0)
#                         supplier_gstin_found = True
#                     elif not structured_data["GSTIN"]:
#                         structured_data["GSTIN"] = match.group(0)
#                 else:
#                     structured_data[key] = match.group(1)

#         if not party_name_detected and not any(re.search(p, clean_line) for p in patterns.values()):
#             structured_data["Party Name"] = clean_line
#             party_name_detected = True
#             continue

#         if party_name_detected and not supplier_name_detected and not any(re.search(p, clean_line) for p in patterns.values()):
#             address_lines.append(clean_line)

#         if any(clean_line.lower().startswith(keyword) for keyword in supplier_keywords):
#             structured_data["Buyer Name"] = clean_line
#             supplier_name_detected = True
#             continue

#         if supplier_name_detected and not supplier_gstin_found and not any(re.search(p, clean_line) for p in patterns.values()):
#             supplier_address_lines.append(clean_line)

#     structured_data["Party Address"] = " ".join(address_lines).strip() if address_lines else None
#     structured_data["Contact Numbers"] = structured_data["Contact Numbers"] if structured_data["Contact Numbers"] else None #check for empty list
#     structured_data["Buyer Address"] = " ".join(supplier_address_lines).strip() if supplier_address_lines else None


import cv2
import numpy as np
import pytesseract
import re
import os
import json
import subprocess

class TextExtractor:
    def __init__(self, image, original_image,table_start_y=None):
        if image is None or original_image is None:
            raise ValueError("Input image or original image is None. Please check the provided images.")
        self.image = image
        self.original_image = original_image
        self.table_start_y = table_start_y
        
        # Crop from top to the starting dimension of the table
        cropped_image = self.image[:self.table_start_y, :]
        
        # Show the cropped image for verification
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)
        
        # Update self.image to the cropped image
        self.image = cropped_image

    def execute(self):
        # Preprocess the image
        self.preprocess_image()
        self.store_processed_image("00LM_preprocessed.jpg", self.image)

        # Perform dilation and contour detection
        self.dilate_image()
        self.store_processed_image("011LM_dilated.jpg", self.dilated_image)

        self.find_contours()
        self.store_processed_image("022LM_contours.jpg", self.image_with_contours_drawn)

        self.convert_contours_to_bounding_boxes()
        self.store_processed_image("033LM_bounding_boxes.jpg", self.image_with_all_bounding_boxes)

        # Process bounding boxes
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.group_bounding_boxes_into_rows()
        self.sort_rows_by_x_coordinate()

        # Crop bounding boxes, perform OCR, and exclude table data
        self.crop_bounding_boxes_and_perform_ocr()
        self.generate_csv_file()
        self.generate_json_response()
        self.print_table_to_console()

        # Extract text using OCR
        text_data    = self.extract_text()
        clean_data   = self.postprocess_text(text_data)
        return clean_data

    def preprocess_image(self):
        """Convert image to grayscale and apply adaptive thresholding."""
        gray       = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        blur=cv2.GaussianBlur(gray,(5,5),3)
        self.image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
        cv2.imshow("img thresh",self.image)
        cv2.waitKey(0)

    def dilate_image(self):
        """Apply dilation to emphasize text regions."""
        kernel             = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.dilated_image = cv2.dilate(self.image, kernel, iterations=3)
        

    def find_contours(self):
        """Detect contours and draw them on a copy of the original image."""
        contours, _   = cv2.findContours(self.dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, contours, -1, (0, 255, 0), 2)

    def convert_contours_to_bounding_boxes(self):
        """Convert contours into bounding boxes and draw them on the image."""
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()

        for contour in self.contours:
            x, y, w, h   = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if h > 10 and w > 10 and 0.2 < aspect_ratio < 10:  # Filter non-text regions
                self.bounding_boxes.append((x, y, w, h))
                cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def get_mean_height_of_bounding_boxes(self):
        """Calculate the average height of all bounding boxes."""
        return np.mean([h for _, _, _, h in self.bounding_boxes])

    def sort_bounding_boxes_by_y_coordinate(self):
        """Sort bounding boxes by their vertical position (y-coordinate)."""
        self.bounding_boxes.sort(key=lambda box: box[1])

    def group_bounding_boxes_into_rows(self):
        """Group bounding boxes into rows based on their y-coordinates."""
        self.rows   = []
        current_row = [self.bounding_boxes[0]]
        threshold   = self.mean_height / 2

        for box in self.bounding_boxes[1:]:
            if abs(box[1] - current_row[-1][1]) <= threshold:
                current_row.append(box)
            else:
                self.rows.append(current_row)
                current_row = [box]

        self.rows.append(current_row)

    def sort_rows_by_x_coordinate(self):
        """Sort bounding boxes within each row by their x-coordinates."""
        for row in self.rows:
            row.sort(key=lambda box: box[0])

    def crop_bounding_boxes_and_perform_ocr(self):
        """Crop each bounding box, perform OCR, and filter table regions."""
        self.table          = []
        image_number        = 0
        table_start_markers = ["Sr.No", "SI.No", "Sr.", "Sr NO"]  # Define table start markers
        table_detected      = False

        for row in self.rows:
            current_row = []
            for x, y, w, h in row:
                cropped_image    = self.original_image[y:y + h, x:x + w]
                image_slice_path = f"./ocr_slices/img_{image_number}.jpg"

                os.makedirs("./ocr_slices", exist_ok=True)
                cv2.imwrite(image_slice_path, cropped_image)

                ocr_result = self.get_result_from_tesseract(image_slice_path)

                # Check if the OCR result indicates the start of a table
                if any(marker.lower() in ocr_result.lower() for marker in table_start_markers):
                    table_detected = True
                    break

                if not table_detected:
                    current_row.append(ocr_result)

                image_number += 1

            if not table_detected and current_row:
                self.table.append(current_row)

    def get_result_from_tesseract(self, image_path):
        """Run Tesseract OCR on an image and return the result."""
        cmd    = f'tesseract {image_path} - -l eng --oem 3 --psm 6'
        output = subprocess.getoutput(cmd)
        return output.strip()

    def generate_csv_file(self):
        """Save OCR results to a CSV file."""
        with open("output_LM11.csv", "w") as file:
            for row in self.table:
                file.write(",".join(row) + "\n")

    def generate_json_response(self):
        """Save OCR results to a JSON file."""
        with open("output_LM11.json", "w") as json_file:
            json.dump(self.table, json_file, indent=4)

    def print_table_to_console(self):
        """Print OCR results in a table format."""
        print("\nExtracted Table Data:")
        for row in self.table:
            print(", ".join(row))

    def store_processed_image(self, file_name, image):
        """Save intermediate processing images for debugging."""
        path = f"./AksharIspat/text_extractor/{file_name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    def extract_text(self):
        """Extract text from the entire preprocessed image using Tesseract."""
        custom_config = r'--oem 3 --psm 6'
        raw_text      = pytesseract.image_to_string(self.image, config=custom_config)

        # Exclude text related to the table section
        table_start_markers = ["Sr.No", "SI.No", "Sr.", "Sr NO"]
        filtered_lines      = []
        table_detected      = False

        for line in raw_text.split("\n"):
            if any(marker.lower() in line.lower() for marker in table_start_markers):
                table_detected = True
                break
            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def postprocess_text(self, text_data):
        """Clean up the extracted text data."""
        lines         = text_data.split("\n")
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(cleaned_lines) 

import re
#GEMINI
def extract_key_data_from_text(extracted_text):
    """Extract and structure key data, handling multiple contact numbers per line."""
    structured_data = {
        "Party Name"     : None,
        "Party Address"  : None,
        "Contact Numbers": [],
        "GSTIN": None,
        "PAN"  : None,
        "Invoice No"  : None,
        "Invoice Date": None,
        "State Code"  : None,
        "State"          : None,
        "Place of Supply": None,
        "Buyer Name"   : None,
        "Buyer Address": None,
        "Buyer GSTIN"  : None
    }

    patterns = {
        "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
        "PAN"  : r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",
        "Phone": r"\b\d{10}\b",
        "Invoice No"  : r"Invoice No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
        "Invoice Date": r"Invoice Date\:\s*(\d{2}/\d{2}/\d{4})",
        "State Code"  : r"State Code\s*[:\-]?\s*(\d+)",
        "State"          : r"State\s*[:\-]?\s*([A-Za-z\s]+)",
        "Place of Supply": r"Place of Supply\s*[:\-]?\s*(\d+)\s*\-\s*([A-Za-z\s]+)"
    }

    supplier_keywords = ["to", "m/s", "mr.", "mrs.", "M/s."]
    lines             = extracted_text.split("\n")

    party_name_detected    = False
    address_lines          = []
    supplier_name_detected = False
    supplier_address_lines = []
    supplier_gstin_found   = False

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Extract multiple phone numbers from a single line
        phone_matches = re.findall(patterns["Phone"], clean_line)
        structured_data["Contact Numbers"].extend(phone_matches)  # Use extend to add multiple

        for key, pattern in patterns.items():
            if key == "Phone": #skip phone because it is already handled above
                continue

            match = re.search(pattern, clean_line)
            if match:
                if key == "Place of Supply":
                    structured_data[key] = {"Code": match.group(1), "State": match.group(2).strip()}
                elif key == "GSTIN":
                    if supplier_name_detected and not supplier_gstin_found:
                        structured_data["Supplier GSTIN"] = match.group(0)
                        supplier_gstin_found = True
                    elif not structured_data["GSTIN"]:
                        structured_data["GSTIN"] = match.group(0)
                else:
                    structured_data[key] = match.group(1)

        if not party_name_detected and not any(re.search(p, clean_line) for p in patterns.values()):
            structured_data["Party Name"] = clean_line
            party_name_detected = True
            continue

        if party_name_detected and not supplier_name_detected and not any(re.search(p, clean_line) for p in patterns.values()):
            address_lines.append(clean_line)

        if any(clean_line.lower().startswith(keyword) for keyword in supplier_keywords):
            structured_data["Buyer Name"] = clean_line
            supplier_name_detected = True
            continue

        if supplier_name_detected and not supplier_gstin_found and not any(re.search(p, clean_line) for p in patterns.values()):
            supplier_address_lines.append(clean_line)

    structured_data["Party Address"] = " ".join(address_lines).strip() if address_lines else None
    structured_data["Contact Numbers"] = structured_data["Contact Numbers"] if structured_data["Contact Numbers"] else None #check for empty list
    structured_data["Buyer Address"] = " ".join(supplier_address_lines).strip() if supplier_address_lines else None

    return structured_data



#==============================================================    
#|                                                            |
#|                       Important Codes Above                |
#|                                                            |
#==============================================================



# import cv2
# import os
# import pytesseract
# from img2table.document import Image
# from datetime import datetime
# import numpy as np

# class TextExtractor:
#     def __init__(self,original_image, image, table_start_y=None):
#         if image is None:
#             raise ValueError("Input image is None.")
        
#         # Crop the image if table_start_y is provided
#         self.original_image=original_image
#         self.image = image if table_start_y is None else image[:table_start_y+20, :]
#         self.table_start_y = table_start_y

#         # Create base output folder if it doesn't exist
#         self.base_output_folder = "BOX-LOGIC-RECTS"
#         if not os.path.exists(self.base_output_folder):
#             os.makedirs(self.base_output_folder)

#         # Generate a unique folder name for this run (timestamp based)
#         self.unique_folder = self.create_unique_folder()

#     def create_unique_folder(self):
#         """Create a new folder with a unique name based on the current timestamp."""
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         new_folder_path = os.path.join(self.base_output_folder, timestamp)
#         if not os.path.exists(new_folder_path):
#             os.makedirs(new_folder_path)
#         return new_folder_path

#     def execute(self):
#         # Detect and process rectangles using img2table first
#         detected = self.detect_rectangles_with_img2table()

#         # If no rectangles were found, fallback to OpenCV rectangle detection
#         if not detected:
#             print("No rectangles detected using img2table, attempting OpenCV method...")
#             self.detect_rectangles_with_opencv()

#     def save_image_temp(self, img):
#         """Save the image to a temporary path and return the path."""
#         temp_path = os.path.join(self.unique_folder, "temp_image.png")
#         cv2.imwrite(temp_path, img)
#         return temp_path

#     def detect_rectangles_with_img2table(self):
#         """Use img2table to detect and process rectangles."""
#         # Save the cropped image temporarily
#         temp_path = self.save_image_temp(self.image)

#         # Instantiate the Image object from img2table
#         img = Image(src=temp_path)

#         # Extract tables from the image
#         image_tables = img.extract_tables()

#         # Initialize a set to store unique bounding boxes
#         detected_boxes = []

#         # Process each detected table
#         for table_index, table in enumerate(image_tables):
#             print(f"Processing Table {table_index + 1}...")

#             for id_row, row in enumerate(table.content.values()):
#                 for id_col, cell in enumerate(row):
#                     # Get bounding box coordinates
#                     x1, y1, x2, y2 = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
#                     new_box = (x1, y1, x2, y2)

#                     # Check for duplicate boxes based on IoU (Intersection over Union)
#                     if self.is_unique_box(new_box, detected_boxes):
#                         detected_boxes.append(new_box)

#                         # Crop the rectangle (region of interest)
#                         roi = self.image[y1:y2, x1:x2]

#                         # Save the cropped rectangle as an image in the unique folder
#                         filename = f"{self.unique_folder}/table_{table_index + 1}_cell_{id_row + 1}_{id_col + 1}.png"
#                         cv2.imwrite(filename, roi)

#                         # Optional: Use Tesseract to extract text (if needed)
#                         value = pytesseract.image_to_string(roi, config='--psm 6')
#                         print(f"Row {id_row + 1}, Col {id_col + 1}: {value.strip() if value.strip() else 'Not Available'}")
#                         print(f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")

#         # If rectangles were detected, return True
#         if detected_boxes:
#             print("Rectangles detected using img2table.")
#             return True

#         # No rectangles detected, return False
#         return False

#     def is_unique_box(self, new_box, existing_boxes, iou_threshold=0.3):
#         """Check if the new_box is unique by comparing it with existing boxes using IoU."""
#         x1_new, y1_new, x2_new, y2_new = new_box

#         for box in existing_boxes:
#             x1_exist, y1_exist, x2_exist, y2_exist = box

#             # Calculate the intersection rectangle
#             x1_inter = max(x1_new, x1_exist)
#             y1_inter = max(y1_new, y1_exist)
#             x2_inter = min(x2_new, x2_exist)
#             y2_inter = min(y2_new, y2_exist)

#             # Calculate intersection and union areas
#             inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
#             area_new = (x2_new - x1_new) * (y2_new - y1_new)
#             area_exist = (x2_exist - x1_exist) * (y2_exist - y1_exist)
#             union_area = area_new + area_exist - inter_area

#             # Compute IoU (Intersection over Union)
#             iou = inter_area / union_area if union_area != 0 else 0

#             # If IoU is above the threshold, the box is not unique
#             if iou > iou_threshold:
#                 return False

#         return True

#     def detect_rectangles_with_opencv(self):
#         """Detect rectangles using OpenCV (contours)."""
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian Blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (7, 7), 5)

#         # Apply Canny edge detection
#         edges = cv2.Canny(blurred, 50, 150)

#         # Use a smaller kernel for dilation to only merge lines without affecting too much of the image
#         kernel = np.ones((3, 3), np.uint8)

#         # Apply dilation to edges only to merge broken lines
#         dilated_edges = cv2.dilate(edges, kernel, iterations=3)

#         # Find contours in the dilated edges
#         contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # Initialize the rectangle ID counter
#         rectangle_id = 1

#         # Iterate over each contour
#         for contour in contours:
#             # Calculate the area of the contour
#             area = cv2.contourArea(contour)

#             # Only process contours with an area larger than 12,000 pixels
#             if area > 16000:
#                 # Approximate the contour with a polygon (4 vertices)
#                 epsilon = 0.04 * cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, epsilon, True)

#                 # Filter for rectangles (4 points)
#                 if len(approx) == 4:
#                     # Get the rotated bounding box
#                     rect = cv2.minAreaRect(contour)
#                     box = cv2.boxPoints(rect)
#                     box = np.int0(box)

#                     # Draw the rotated rectangle
#                     cv2.drawContours(self.image, [box], -1, (0, 255, 0), 2)

#                     # Label the rectangle with its ID and angle
#                     x, y, w, h = cv2.boundingRect(box)
#                     cv2.putText(self.image, f"ID: {rectangle_id}", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#                     # Crop the image around the rotated rectangle
#                     cropped_rect = self.image[y:y + h, x:x + w]

#                     # Save the cropped rectangle as a new image
#                     self.store_processed_image(f"{self.unique_folder}/cropped_rectangle_{rectangle_id}.png", cropped_rect)

#                     # Increment the rectangle ID
#                     rectangle_id += 1

#     def store_processed_image(self, filename, image):
#         """Save the processed image."""
#         cv2.imwrite(filename, image)