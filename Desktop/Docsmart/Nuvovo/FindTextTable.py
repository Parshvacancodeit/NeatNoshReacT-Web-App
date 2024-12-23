# import cv2
# import pytesseract
# import re

# class MainTableExtractor:
#     def __init__(self, img):
#         self.image = img

#     def execute(self):
#         # Convert image to grayscale
#         self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         self.blurred = cv2.GaussianBlur(self.grey, (5, 5), 2)

#         # Apply adaptive thresholding
#         self.thresholded_image = cv2.adaptiveThreshold(
#             self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
#         )

#         # Invert the image to improve OCR accuracy
#         self.inverted_image = cv2.bitwise_not(self.thresholded_image)

#         # Find contours and hierarchy
#         contours, hierarchy = cv2.findContours(self.inverted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#         # Debugging: Display the number of contours
#         print(f"Number of contours found: {len(contours)}")

#         # Extract contour hierarchy and process it to identify parent-child relationships
#         parent_contours = []
#         child_contours = []

#         # Loop through the contours and hierarchy
#         for idx, contour in enumerate(contours):
#             if hierarchy[0][idx][3] == -1:
#                 # This is a parent contour (top-level contour)
#                 parent_contours.append(contour)
#             else:
#                 # This is a child contour (inside a parent contour)
#                 child_contours.append(contour)

#         # Debugging: Show parent and child contours
#         print(f"Parent Contours: {len(parent_contours)}")
#         print(f"Child Contours: {len(child_contours)}")

#         # Optionally, draw contours for visualization
#         for contour in parent_contours:
#             cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)  # Green for parent contours
#         for contour in child_contours:
#             cv2.drawContours(self.image, [contour], -1, (0, 0, 255), 2)  # Red for child contours

#         # Show image with contours drawn
#         cv2.imshow('Contours', self.image)
#         cv2.waitKey(0)

#         # Use pytesseract to extract detailed data
#         details = pytesseract.image_to_data(self.inverted_image, output_type=pytesseract.Output.DICT)

#         # Debugging: Print out the entire OCR captured text
#         print("OCR Captured Text:")
#         for i, word in enumerate(details['text']):
#             if word.strip():  # Ignore empty strings
#                 print(f"{i}: {word} (Left: {details['left'][i]}, Top: {details['top'][i]}, Width: {details['width'][i]}, Height: {details['height'][i]})")

#         # List of header keywords (expanded)
#         header_keywords = [
#             "GST No", "Invoice No", "Description of Goods", "Type", "Quantity",
#             "HSN/SAC", "GST Rate", "Rate", "CGST", "IGST", "SGST", "Amount",
#             "Taxable Value", "Total Tax Amount", "ROUND OFF", "GRAND TOTAL",
#             "Net Amount Payable", "E-Way Bill No", "Product Description",
#             "HSN Code", "Rate Rs", "Unit Cost INR", "Dis%", "Unit", "Party GST No",
#             "HSN Code", "Taxable Amt", "HSN CORD", "Unit", "Price",
#             "Qty", "Qty (Nos)", "Total Inv.Amt", "Document Total", "Total Tax",
#             "Invoice Amount", "Total Before Tax", "C GST", "S GST", "Unit Price"
#         ]

#         matches = []

#         # Look for header keywords in OCR results
#         for i, word in enumerate(details['text']):
#             if word.strip():  # Ignore empty strings
#                 for keyword in header_keywords:
#                     if re.search(r'\b' + re.escape(keyword) + r'\b', word, re.IGNORECASE):
#                         # Add detected word's bounding box info
#                         x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
#                         matches.append((word, x, y, x + w, y + h))

#         # Debugging: Print out the detected keywords and their bounding boxes
#         print(f"Detected Keywords: {matches}")

#         # Remove "SI No" or "Sr" entries from the matches
#         filtered_matches = [match for match in matches if not re.match(r'(SI|Sr|S\.?I)', match[0], re.IGNORECASE)]

#         # Debugging: Print out the filtered matches
#         print(f"Filtered Matches (No SI/Sr): {filtered_matches}")

#         # If no keywords found, raise an error
#         if not filtered_matches:
#             print("No valid header keywords found.")
#             return None

#         # Expand the bounding box to cover the entire table
#         min_x = min([match[1] for match in filtered_matches])
#         min_y = min([match[2] for match in filtered_matches])
#         max_x = max([match[3] for match in filtered_matches])
#         max_y = max([match[4] for match in filtered_matches])

#         # Add some padding to the bounding box to ensure full table is captured
#         padding = 20
#         min_x -= padding
#         min_y -= padding
#         max_x += padding
#         max_y += padding

#         # Make sure the coordinates are within the image bounds
#         min_x = max(min_x, 0)
#         min_y = max(min_y, 0)
#         max_x = min(max_x, self.image.shape[1])
#         max_y = min(max_y, self.image.shape[0])

#         # Crop the main table area using the expanded bounding box
#         main_table_image = self.image[min_y:max_y, min_x:max_x]

#         # Debugging: Display cropped main table area
#         cv2.imshow('Main Table', main_table_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         return main_table_image  # Return the cropped main table area

#LOGIC FOR ENTIRE TABLE DETECTION , CAN DETECT SAME HEADER MULTIPLE TIMES

# import cv2
# import pytesseract
# import re
# import os

# class MainTableExtractor:
#     def __init__(self, img, save_dir='detected_headers'):
#         self.image = img
#         self.save_dir = save_dir  # Directory to save images with drawn bounding boxes
#         if not os.path.exists(save_dir):  # Create the directory if it doesn't exist
#             os.makedirs(save_dir)

#     def execute(self):
#         # Convert image to grayscale
#         self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         self.blurred = cv2.GaussianBlur(self.grey, (5, 5), 2)

#         # Apply adaptive thresholding
#         self.thresholded_image = cv2.adaptiveThreshold(
#             self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
#         )

#         # Invert the image to improve OCR accuracy
#         self.inverted_image = cv2.bitwise_not(self.thresholded_image)

#         # Set the PSM and OEM for OCR, optimal for detecting sparse text
#         custom_oem_psm_config = r'--oem 3 --psm 6'

#         # Use pytesseract to extract detailed data with the proper PSM and OEM
#         details = pytesseract.image_to_data(self.inverted_image, output_type=pytesseract.Output.DICT, config=custom_oem_psm_config)

#         # List of header keywords (including variations for SI/SR)
#         header_keywords = [
#             "si", "sr","sj" "sr. no", "si. no", "serial no", "item", "description", "quantity", "amount",
#             "gst no", "invoice no", "description of goods", "type", "hsn/sac", "gst rate", "rate", 
#             "e-way bill no", "product", "hsn",
#             "particular", "sr.", "assessable value",
#             "product description", "description of goods", "product name & desc", "particulars", "description of goods /services",
#             "description", "product name", "description of goods and services", "particulars/items",
#             "per", "unit", "per", "uqc",
#             "hsn/sac code", "hsn code", "hsn/sac", "hsn"
#         ]

#         matches = []

#         # Look for header keywords in OCR results
#         for i, word in enumerate(details['text']):
#             if word.strip():  # Ignore empty strings
#                 for keyword in header_keywords:
#                     if re.search(r'\b' + re.escape(keyword) + r'\b', word.lower(), re.IGNORECASE):  # Compare both in lowercase
#                         # Add detected word's bounding box info
#                         x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
#                         matches.append((word, x, y, x + w, y + h))

#         # Separate SI/SR related matches to appear at the start
#         si_sr_keywords = ['si', 'sr', 'si. no', 'sr. no', 'serial no', 'no.']
#         si_sr_matches = [match for match in matches if any(re.search(r'\b' + re.escape(keyword) + r'\b', match[0].lower(), re.IGNORECASE) for keyword in si_sr_keywords)]
        
#         # Other matches will come after SI/SR keywords
#         other_matches = [match for match in matches if match not in si_sr_matches]

#         # Combine the lists to put SI/SR matches first
#         sorted_matches = si_sr_matches + other_matches

#         # If no keywords found, raise an error
#         if not sorted_matches:
#             print("No valid header keywords found.")
#             return None

#         # Create a copy of the original image to draw on
#         image_copy = self.image.copy()

#         # Draw bounding boxes around the detected header keywords on the copied image
#         for match in sorted_matches:
#             word, x1, y1, x2, y2 = match
#             cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle

#             # Optionally, add the text label to the bounding box
#             cv2.putText(image_copy, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Generate the output file path based on the input image name
#         image_name = "output_image"  # Default name if not specified
#         output_image_path = os.path.join(self.save_dir, f"{image_name}_detected.jpg")

#         # Save the image with bounding boxes drawn around detected headers (on the copy)
#         cv2.imwrite(output_image_path, image_copy)

#         # Debugging: Print out where the image is saved
#         print(f"Image with detected headers saved to: {output_image_path}")

#         # Optionally, display the image for debugging purposes
#         cv2.imshow('Detected Headers', image_copy)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         # Define the bounding box that encloses the main table
#         min_x = min([match[1] for match in sorted_matches])
#         min_y = min([match[2] for match in sorted_matches])
#         max_x = max([match[3] for match in sorted_matches])
#         max_y = max([match[4] for match in sorted_matches])

#         # Expand the bounding box to cover the entire table
#         padding = 170
#         padding2 =0
#           # Adjust padding to ensure the entire table is included
#         min_x -= padding
#         min_y += padding2
#         max_x += padding
#         max_y -= padding2

#         # Make sure the coordinates are within the image bounds
#         min_x = max(min_x, 0)
#         min_y = max(min_y, 0)
#         max_x = min(max_x, self.image.shape[1])
#         max_y = min(max_y, self.image.shape[0])

#         # Crop the main table area using the expanded bounding box
#         main_table_image = self.image[min_y:max_y, min_x:max_x]

#         return main_table_image  # Return the cropped main table area


#Logic for detecting entire axis ignore it for now


# import cv2
# import pytesseract
# import re
# import os

# class MainTableExtractor:
#     def __init__(self, img, save_dir='detected_headers'):
#         self.image = img
#         self.save_dir = save_dir  # Directory to save images with drawn bounding boxes
#         if not os.path.exists(save_dir):  # Create the directory if it doesn't exist
#             os.makedirs(save_dir)

#     def execute(self):
#         # Convert image to grayscale
#         self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         self.blurred = cv2.GaussianBlur(self.grey, (5, 5), 2)

#         # Apply adaptive thresholding
#         self.thresholded_image = cv2.adaptiveThreshold(
#             self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
#         )

#         # Invert the image to improve OCR accuracy
#         self.inverted_image = cv2.bitwise_not(self.thresholded_image)

#         # Set the PSM and OEM for OCR, optimal for detecting sparse text
#         custom_oem_psm_config = r'--oem 3 --psm 6'

#         # Use pytesseract to extract detailed data with the proper PSM and OEM
#         details = pytesseract.image_to_data(self.inverted_image, output_type=pytesseract.Output.DICT, config=custom_oem_psm_config)

#         # List of header keywords
#         header_keywords = [
#             "si", "sr", "sj", "sr. no", "si. no", "serial no", "item", "description", "quantity", "amount",
#             "gst no", "invoice no", "description of goods", "type", "hsn/sac", "gst rate", "rate", 
#             "e-way bill no", "product", "hsn",
#             "particular", "sr.", "assessable value",
#             "product description", "description of goods /services",
#             "description", "product name", "per", "unit", "uqc",
#             "hsn/sac code", "hsn code", "hsn/sac", "hsn"
#         ]

#         matches = []
#         found_keywords = set()  # Track keywords already found

#         # Detect keywords and their bounding boxes
#         for i, word in enumerate(details['text']):
#             if word.strip():  # Ignore empty strings
#                 for keyword in header_keywords:
#                     if re.search(r'\b' + re.escape(keyword) + r'\b', word.lower(), re.IGNORECASE):
#                         if keyword not in found_keywords:
#                             # Add detected word's bounding box info
#                             x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
#                             matches.append((word, x, y, x + w, y + h))
#                             found_keywords.add(keyword)

#         # Check if matches are found
#         if not matches:
#             print("No valid header keywords found.")
#             return None

#         # Sort matches based on position
#         matches = sorted(matches, key=lambda x: (x[2], x[1]))  # Sort by Y first, then by X

#         # Create a copy of the original image to draw on
#         image_copy = self.image.copy()

#         # Draw bounding boxes around the detected header keywords on the copied image
#         for match in matches:
#             word, x1, y1, x2, y2 = match
#             cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
#             cv2.putText(image_copy, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Add text label

#         # Define the bounding box for the table
#         min_y = min([match[2] for match in matches])
#         max_y = max([match[4] for match in matches])

#         # Ensure the width spans the entire image
#         min_x = 0
#         max_x = self.image.shape[1]

#         # Calculate height and width of the table
#         table_height = max_y - min_y
#         table_width = max_x - min_x

#         # Validate the table size
#         if table_height < 100 or table_width < 400:
#             print(f"Table detected but does not meet minimum size criteria: Height={table_height}, Width={table_width}")
#             return None

#         # Crop the table from the image
#         table_image = self.image[min_y:max_y, min_x:max_x]

#         # Save the cropped table image
#         output_image_path = os.path.join(self.save_dir, "filtered_table_image.jpg")
#         cv2.imwrite(output_image_path, table_image)

#         print(f"Filtered table image saved to: {output_image_path}")

#         # Show the cropped table image
#         cv2.imshow('Table Image', table_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         return table_image  # Return the cropped main table area



#code with proper table detection not accurate in header detection

import cv2
import pytesseract
import re
import os

class FindTextTable:
    def __init__(self, img, save_dir='detected_headers'):
        self.image = img
        self.save_dir = save_dir  # Directory to save images with drawn bounding boxes
        if not os.path.exists(save_dir):  # Create the directory if it doesn't exist
            os.makedirs(save_dir)

    def execute(self):
        # Convert image to grayscale
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        self.blurred = cv2.GaussianBlur(self.grey, (5, 5), 6)

        # Apply adaptive thresholding
        self.thresholded_image = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8
        )

        # Invert the image to improve OCR accuracy
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

        # Set the PSM and OEM for OCR, optimal for detecting sparse text
        custom_oem_psm_config = r'--oem 3 --psm 6'

        # Use pytesseract to extract detailed data with the proper PSM and OEM
        details = pytesseract.image_to_data(self.inverted_image, output_type=pytesseract.Output.DICT, config=custom_oem_psm_config)

        # List of header keywords
        header_keywords = [
            "item", "description", "quantity", "amount",
            "gst no", "invoice no", "description of goods", "hsn/sac", "gst rate", "rate",
            "e-way bill no", "product", "hsn", "particular", "assessable value",
            "sr.", "product description", "description of goods /services",
            "description", "product name", "per", "unit", "uqc",
            "hsn/sac code", "hsn code", "hsn/sac", "hsn", "grade", "material code", "uom", "hsn code", "rate in ₹", "amount in ₹"
        ]

        # Store detected header keyword positions
        header_positions = []
        found_keywords = set()  # To avoid duplicate keywords

        # Detect keywords and their bounding boxes
        for i, word in enumerate(details['text']):
            if word.strip():  # Ignore empty strings
                for keyword in header_keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', word.lower(), re.IGNORECASE):
                        if keyword not in found_keywords:
                            # Record bounding box for header
                            x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
                            header_positions.append((keyword, x, y, x + w, y + h))
                            found_keywords.add(keyword)

        # If no valid headers found, return
        if not header_positions:
            print("No valid header keywords found.")
            return None

        # Sort the header positions by their y-coordinate to get the first row
        header_positions.sort(key=lambda x: x[2])  # Sort by the top position (y-coordinate)

        # Create a copy of the image for visualization
        image_copy = self.image.copy()

        # Draw bounding boxes around the detected header keywords on the copied image
        for word, x1, y1, x2, y2 in header_positions:
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
            cv2.putText(image_copy, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Add text label
            print(f"Detected Header: '{word}' at Position: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Save the image with highlighted headers
        output_image_path = os.path.join(self.save_dir, "detected_headers.jpg")
        cv2.imwrite(output_image_path, image_copy)

        print(f"Image with detected headers saved to: {output_image_path}")

        # Display the image with detected headers
        cv2.imshow('Detected Headers', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Calculate the table's bounding box
        min_y = min([y1 for _, _, y1, _, _ in header_positions])
        max_y = max([y2 for _, _, _, _, y2 in header_positions])

        # Use line detection to refine the table region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(self.thresholded_image, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_y = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

        # Define the starting point for the table region: from the header row downwards
        min_x = 0
        max_x = self.image.shape[1]

        # Extend the region to include all rows below the header dynamically
        table_region = self.image[min_y:max_y, min_x:max_x]  # Removed fixed padding for dynamic table extraction

        # Save the cropped table image
        output_table_path = os.path.join(self.save_dir, "filtered_table_image.jpg")
        cv2.imwrite(output_table_path, table_region)

        print(f"Filtered table image saved to: {output_table_path}")

        return min_y  # Return the cropped main table area
    