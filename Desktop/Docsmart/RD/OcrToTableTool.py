
# import cv2
# import numpy as np
# import subprocess
# import os
# import json

# class OcrToTableTool:
#     def __init__(self, image, original_image):
#         """
#         Initialize the OCR tool with a thresholded image and the original image.
#         """
#         self.thresholded_image = image
#         self.original_image = original_image
#         self.headers = ['Sr.', 'Particular', 'HSN', 'Quantity', 'Unit', 'Rate', 'Assessable Value', 'GST %', 'CGST', 'SGST', 'Amount']

#     def execute(self):
#         """
#         Execute the full OCR process: dilation, contour detection, bounding box creation, OCR, and CSV/JSON generation.
#         """
#         self.dilate_image()
#         self.store_process_image('0_dilated_image.jpg', self.dilated_image)

#         self.find_contours()
#         self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)

#         self.convert_contours_to_bounding_boxes()
#         self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)

#         self.mean_height = self.get_mean_height_of_bounding_boxes()
#         self.sort_bounding_boxes_by_y_coordinate()

#         self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
#         self.sort_all_rows_by_x_coordinate()

#         self.crop_each_bounding_box_and_ocr()
#         self.generate_csv_file()
#         self.generate_json_response()
#         self.print_table_to_console()

#     def print_table_to_console(self):
#         """Print the extracted table data to the console."""
#         print("\nExtracted Table Data:")
#         for row in self.table:
#             print(", ".join(row))

#     def dilate_image(self):
#         # Smaller kernel to remove gaps but avoid merging words
#         kernel = np.ones((3, 4), np.uint8)
#         self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=5)
        
#         # Optional: Apply opening (erosion + dilation) to remove small noise
#         kernel_for_opening = np.ones((2, 3), np.uint8)
#         self.dilated_image = cv2.morphologyEx(self.dilated_image, cv2.MORPH_OPEN, kernel_for_opening)

#         # Apply closing to refine gaps between words
#         kernel_for_closing = np.ones((3, 3), np.uint8)
#         self.dilated_image = cv2.morphologyEx(self.dilated_image, cv2.MORPH_CLOSE, kernel_for_closing)

#         # Show the image to check the effect of dilation and morphological operations
#         # cv2.imshow("Dilated Image", self.dilated_image)
#         # cv2.waitKey(0)

#     def find_contours(self):
#         """
#         Detect contours from the dilated image.
#         """
#         contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         self.contours = contours
#         self.image_with_contours_drawn = self.original_image.copy()
#         cv2.drawContours(self.image_with_contours_drawn, contours, -1, (0, 255, 0), 3)
#         # cv2.imshow("contour Image", self.image_with_contours_drawn)
#         # cv2.waitKey(0)
        
        


#     def convert_contours_to_bounding_boxes(self):
#         """
#         Convert contours to bounding boxes and draw them on the image.
#         """
#         self.bounding_boxes = []
#         self.image_with_all_bounding_boxes = self.original_image.copy()
#         for contour in self.contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             self.bounding_boxes.append((x, y, w, h))
#             cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # cv2.imshow("contour Image", self.image_with_all_bounding_boxes)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
        

#     def get_mean_height_of_bounding_boxes(self):
#         """
#         Calculate the mean height of all bounding boxes.
#         """
#         return np.mean([h for _, _, _, h in self.bounding_boxes])

#     def sort_bounding_boxes_by_y_coordinate(self):
#         """
#         Sort bounding boxes by their y-coordinate (vertical position).
#         """
#         self.bounding_boxes.sort(key=lambda x: x[1])

#     def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
#         """
#         Group bounding boxes into rows based on their y-coordinate proximity.
#         """
#         self.rows = []
#         current_row = [self.bounding_boxes[0]]
#         threshold = self.mean_height / 2

#         for box in self.bounding_boxes[1:]:
#             if abs(box[1] - current_row[-1][1]) <= threshold:
#                 current_row.append(box)
#             else:
#                 self.rows.append(current_row)
#                 current_row = [box]

#         self.rows.append(current_row)

#     def sort_all_rows_by_x_coordinate(self):
#         """
#         Sort all bounding boxes in each row by their x-coordinate (horizontal position).
#         """
#         for row in self.rows:
#             row.sort(key=lambda x: x[0])

#     def crop_each_bounding_box_and_ocr(self):
#         """
#         Crop each bounding box and perform OCR to extract text.
#         """
#         self.table = []
#         image_number = 0

#         for row in self.rows:
#             current_row = []
#             for x, y, w, h in row:
#                 cropped_image = self.original_image[max(0, y - 5):y + h, x:x + w]
#                 image_slice_path = f"./ocr_slices/img_{image_number}.jpg"

#                 if not os.path.exists("./ocr_slices"):
#                     os.makedirs("./ocr_slices")

#                 cv2.imwrite(image_slice_path, cropped_image)
#                 ocr_result = self.get_result_from_tesseract(image_slice_path)
#                 current_row.append(ocr_result)
#                 image_number += 1

#             self.table.append(current_row)

#     def get_result_from_tesseract(self, image_path):
#         """
#         Run Tesseract OCR on the cropped image.
#         """
#         cmd = f'tesseract {image_path} - -l eng --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()|%.calmg* "'
#         output = subprocess.getoutput(cmd)
#         return output.strip()

#     def generate_csv_file(self):
#         """
#         Save the extracted table data into a CSV file.
#         """
#         with open("output.csv", "w") as file:
#             for row in self.table:
#                 file.write(",".join(row) + "\n")

#     def generate_json_response(self):
#         """
#         Generate the extracted table data in JSON format with proper headers and key-value mapping.
#         """
#         json_response = []
#         for row in self.table:
#             row_data = dict(zip(self.headers, row))
#             json_response.append(row_data)

#         with open("output1.json", "w") as json_file:
#             json.dump(json_response, json_file, indent=4)

#         print("\nGenerated JSON Response:")
#         print(json.dumps(json_response, indent=4))

#     def store_process_image(self, file_name, image):
#         """
#         Save a processed image for debugging or visualization.
#         """
#         path = f"./Bholenath/ocr_table_tool_bh/{file_name}"

#         if not os.path.exists("./Bholenath/ocr_table_tool_bh/"):
#             os.makedirs("./Bholenath/ocr_table_tool_bh/")

#         cv2.imwrite(path, image)

###==============================Works Perfectly =========================================###
###===========================Gives Each ouput In row and keyvalue Format=================###
############===============No Text Noise removal , ===========================#######
# import cv2
# import numpy as np
# import subprocess
# import os
# import json
# from textblob import Word, TextBlob


# class OcrToTableTool:
#     def __init__(self, image, original_image):
#         self.thresholded_image = image
#         self.original_image = original_image
#         self.headers = ['Sr.', 'Product Description', 'HSN/SAC', 'Quantity', 'Rate', 'Per', 'Amount']

#     def execute(self):
#         self.dilate_image()
#         self.store_process_image('0_dilated_image.jpg', self.dilated_image)

#         self.find_contours()
#         self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)

#         self.convert_contours_to_bounding_boxes()
#         self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)

#         self.mean_height = self.get_mean_height_of_bounding_boxes()
#         self.sort_bounding_boxes_by_y_coordinate()

#         self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
#         self.sort_all_rows_by_x_coordinate()

#         self.crop_each_bounding_box_and_ocr()
#         self.generate_csv_file()
#         self.generate_json_response()
#         self.print_table_to_console()

#     def print_table_to_console(self):
#         print("\nExtracted Table Data:")
#         for index, row in enumerate(self.table):
#             print(f"Row {index + 1}: " + ", ".join(row))

#     def dilate_image(self):
#         kernel = np.ones((2, 5), np.uint8)
#         self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=5)

#     def find_contours(self):
#         self.image_with_contours_drawn = self.original_image.copy()
#         contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         self.contours = [contour for contour in contours if cv2.contourArea(contour) > 700 ]
#         cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 2)

#     def convert_contours_to_bounding_boxes(self):
#         self.bounding_boxes = []
#         self.image_with_all_bounding_boxes = self.original_image.copy()
#         for contour in self.contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             self.bounding_boxes.append((x, y, w, h))
#             cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     def get_mean_height_of_bounding_boxes(self):
#         return np.mean([h for _, _, _, h in self.bounding_boxes])

#     def sort_bounding_boxes_by_y_coordinate(self):
#         self.bounding_boxes.sort(key=lambda x: x[1])

#     def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
#         self.rows = []
#         current_row = [self.bounding_boxes[0]]
#         threshold = self.mean_height / 2

#         for box in self.bounding_boxes[1:]:
#             if abs(box[1] - current_row[-1][1]) <= threshold:
#                 current_row.append(box)
#             else:
#                 self.rows.append(current_row)
#                 current_row = [box]
#         self.rows.append(current_row)

#     def sort_all_rows_by_x_coordinate(self):
#         for row in self.rows:
#             row.sort(key=lambda x: x[0])

#     def crop_each_bounding_box_and_ocr(self):
#         self.table = []
#         image_number = 0

#         for row_index, row in enumerate(self.rows):
#             current_row = []
#             print(f"\nProcessing Row {row_index + 1}:")
#             for bbox_index, (x, y, w, h) in enumerate(row):
#                 cropped_image = self.original_image[max(0, y - 5):y + h, x:x + w]
#                 slice_path = self.save_image_slice(cropped_image, image_number)

#                 ocr_result = self.get_result_from_tesseract(slice_path)
                
#                 current_row.append(ocr_result)

#                 print(f"  Bounding Box {bbox_index + 1}: OCR Result -> '{ocr_result}'")
#                 image_number += 1
#             self.table.append(current_row)

#     def save_image_slice(self, image, image_number):
#         os.makedirs("./ocr_slices", exist_ok=True)
#         preprocessed_image = self.preprocess_image(image)
#         slice_path = f"./ocr_slices/img_{image_number}.jpg"
#         cv2.imwrite(slice_path, preprocessed_image)
#         return slice_path

#     def preprocess_image(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduces noise
#         thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#         return cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

#     def get_result_from_tesseract(self, image_path):
#         cmd = f'tesseract {image_path} - -l eng --oem 3 --psm 6 --dpi 300'
#         return subprocess.getoutput(cmd).strip()

#     def apply_textblob_correction(self, text):
#         """Correct spelling only if needed, without altering valid words."""
#         corrected_words = []
#         for word in text.split():
#             w = Word(word)
#             suggestions = w.spellcheck()
#             if suggestions[0][0].lower() == word.lower():  # Word is valid
#                 corrected_words.append(word)
#             else:
#                 corrected_words.append(suggestions[0][0])  # Take best suggestion
#         return " ".join(corrected_words)

#     def generate_csv_file(self):
#         with open("output.csv", "w") as file:
#             for row in self.table:
#                 file.write(",".join(row) + "\n")

#     def generate_json_response(self):
#         json_response = []
#         for row in self.table:
#             row_data = dict(zip(self.headers, row))
#             json_response.append(row_data)
#         with open("output1.json", "w") as json_file:
#             json.dump(json_response, json_file, indent=4)
#         print("\nGenerated JSON Response:")
#         print(json.dumps(json_response, indent=4))

#     def store_process_image(self, file_name, image):
#         os.makedirs("./Bholenath/ocr_table_tool_bh/", exist_ok=True)
#         path = f"./Bholenath/ocr_table_tool_bh/{file_name}"
#         cv2.imwrite(path, image)



import cv2
import numpy as np
import subprocess
import os
import json
from textblob import Word, TextBlob

class OcrToTableTool:
    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image
        self.headers = ['Sr.', 'Product Description', 'HSN/SAC', 'Quantity', 'Rate', 'Per', 'Amount'] #,'TMT':'TMT Bars -8 MM' ,,'Bars -8 MM'
        self.text_corrections = {'Description of':'Description of Goods and Services','S|]':'SI.No'}  # Dictionary for storing incorrect and corrected text
        self.noisy_texts = ['Goods and Services','C','Lo','No.','Le','/','aN','p)']  # List for noisy text that should be ignored

    def execute(self):
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)

        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)

        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)

        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()

        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()

        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()
        self.generate_json_response()
        self.print_table_to_console()

    def print_table_to_console(self):
        print("\nExtracted Table Data:")
        for index, row in enumerate(self.table):
            row_data = ", ".join(row)
            if row_data:  # Only print non-empty rows
                print(f"Row {index + 1}: {row_data}")

    def dilate_image(self):
        kernel = np.ones((2, 5), np.uint8)
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=5)

    def find_contours(self):
        self.image_with_contours_drawn = self.original_image.copy()
        contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [contour for contour in contours if cv2.contourArea(contour) > 700 ]
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 2)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def get_mean_height_of_bounding_boxes(self):
        return np.mean([h for _, _, _, h in self.bounding_boxes])

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes.sort(key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        current_row = [self.bounding_boxes[0]]
        threshold = self.mean_height / 2

        for box in self.bounding_boxes[1:]:
            if abs(box[1] - current_row[-1][1]) <= threshold:
                current_row.append(box)
            else:
                self.rows.append(current_row)
                current_row = [box]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        image_number = 0

        for row_index, row in enumerate(self.rows):
            current_row = []
            print(f"\nProcessing Row {row_index + 1}:")
            for bbox_index, (x, y, w, h) in enumerate(row):
                cropped_image = self.original_image[max(0, y - 5):y + h, x:x + w]
                slice_path = self.save_image_slice(cropped_image, image_number)

                ocr_result = self.get_result_from_tesseract(slice_path)
                
                # Apply text correction if needed
                ocr_result = self.apply_correction(ocr_result)

                # Check if the result is noisy, if so, ignore it
                if ocr_result not in self.noisy_texts and ocr_result.strip():
                    current_row.append(ocr_result)
                image_number += 1
            self.table.append(current_row)

    def save_image_slice(self, image, image_number):
        os.makedirs("./ocr_slices", exist_ok=True)
        preprocessed_image = self.preprocess_image(image)
        slice_path = f"./ocr_slices/img_{image_number}.jpg"
        cv2.imwrite(slice_path, preprocessed_image)
        return slice_path

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduces noise
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    def get_result_from_tesseract(self, image_path):
        cmd = f'tesseract {image_path} - -l eng --oem 3 --psm 6 --dpi 300'
        return subprocess.getoutput(cmd).strip()

    def apply_correction(self, text):
        """Corrects text using the text_corrections dictionary if a match is found."""
        for incorrect, corrected in self.text_corrections.items():
            text = text.replace(incorrect, corrected)
        return text

    def generate_csv_file(self):
        with open("output.csv", "w") as file:
            for row in self.table:
                file.write(",".join(row) + "\n")

    def generate_json_response(self):
        json_response = []
        for row in self.table:
            row_data = dict(zip(self.headers, row))
            json_response.append(row_data)
        with open("output1.json", "w") as json_file:
            json.dump(json_response, json_file, indent=4)
        print("\nGenerated JSON Response:")
        print(json.dumps(json_response, indent=4))

    def store_process_image(self, file_name, image):
        os.makedirs("./Bholenath/ocr_table_tool_bh/", exist_ok=True)
        path = f"./Bholenath/ocr_table_tool_bh/{file_name}"
        cv2.imwrite(path, image)

