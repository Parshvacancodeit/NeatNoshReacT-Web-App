import cv2
import numpy as np
import subprocess
import os
import json
from textblob import Word, TextBlob

class TextOcrToTool:
    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image
        self.headers = ['1.', '2', '3', '4', '5', '6', '7','8','9','10','11','12'] #,'TMT':'TMT Bars -8 MM' ,,'Bars -8 MM'
        self.text_corrections = {'Description of':'Description of Goods and Services','S|]':'SI.No','Sr}':'SR.No',"Rate in <":"Rate in ₹"}  # Dictionary for storing incorrect and corrected text
        self.noisy_texts = ['Goods and Services','C','Lo','No.','Le','/','aN','p)','AN','Sf','ig','(/','>']  # List for noisy text that should be ignored

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

    # def print_table_to_console(self):
    #     print("\nExtracted Table Data:")
    #     for index, row in enumerate(self.table):
    #         row_data = ", ".join(row)
    #         if row_data:  # Only print non-empty rows
    #             print(f"Row {index + 1}: {row_data}")

    def dilate_image(self):
        kernel = np.ones((2, 10), np.uint8)
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=3)


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
            cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w-3, y + h-3), (0, 255, 0), 2)

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
        # print("\nGenerated JSON Response:")
        # print(json.dumps(json_response, indent=4))

    def store_process_image(self, file_name, image):
        os.makedirs("./Bholenath/ocr_table_tool_bh/", exist_ok=True)
        path = f"./Bholenath/ocr_table_tool_bh/{file_name}"
        cv2.imwrite(path, image)

    def print_table_to_console(self):
        print("\nExtracted row Data:")
        # Print each row directly
        for index, row in enumerate(self.table):
            print(f"Row {index + 1}: {', '.join(row)}")
