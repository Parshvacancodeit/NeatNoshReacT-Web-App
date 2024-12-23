import cv2
import TextExtractor as tx
import json
import os
import FindTextTable as Tab
import TableExtractor as te
from barcode_remover import BarcodeRemover
import TextLineRemover as tlr
import TextOcrToTool as ottt
import CropTextRegion as ctr


# Path to the input image
path_to_image = "/Users/parshvapatel/Desktop/PS/Data_Extraction_using_OCR/image/AR.jpeg"
folder_path = "/Users/parshvapatel/Desktop/PS/Data_Extraction_using_OCR/output_tables"

try:
    # Load the original image
    table_extractor = te.TableExtractor(path_to_image)
    perspective_corrected_image = table_extractor.execute()
    barcode_remover = BarcodeRemover(perspective_corrected_image)
    final_image = barcode_remover.remove_barcodes()
    # final_image = barcode_remover.handle_vertical_barcodes()
    cv2.imshow("IMG",final_image)
    cv2.waitKey(0)
    
    

    if perspective_corrected_image is None:
        raise ValueError("Failed to load the image. Please check the file path.")
    
    
    text_reg_finder = Tab.FindTextTable(final_image)
    table_start_y = text_reg_finder.execute()


    # Initialize TextExtractor with the original image
    text_reg = ctr.CropTextRegion(image=final_image, original_image=perspective_corrected_image,table_start_y=table_start_y)
    text_reg_img=text_reg.execute()
    lines_remover = tlr.TextLineRemover(text_reg_img) 

     # Use cropped main table
    image_without_lines = lines_remover.execute()
        #cv2.imshow("Image Without Lines", image_without_lines)

        # OCR to extract table data
    ocr_tool = ottt.TextOcrToTool(image_without_lines, text_reg_img)
    ocr_tool.execute()

    # Extract text from the image
    #extracted_text = text_extractor.execute()

    # print("\nExtracted Text from the Whole Page:")
    # print(extracted_text)

    # Extract key structured data from the extracted text
    #structured_data = tx.extract_key_data_from_text(extracted_text)

    # Ensure the output directory exists
    # output_directory = "./AksharIspat/"
    # os.makedirs(output_directory, exist_ok=True)

    # # Save the structured data to a JSON file
    # output_json_path = os.path.join(output_directory, "extracted_data_LEDData_16.json")
    # with open(output_json_path, "w", encoding="utf-8") as json_file:
    #     json.dump(structured_data, json_file, indent=4, ensure_ascii=False)

    # print(f"\nExtracted structured data has been saved in JSON format to {output_json_path}")

except Exception as e:
    print(f"An error occurred: {e}")

# Optional: Display the original image (for debugging or confirmation)
if perspective_corrected_image is not None:
    cv2.imshow("Original Image", perspective_corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()