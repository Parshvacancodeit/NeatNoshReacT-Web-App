# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2

# path_to_image               = "./Main_Ocr/LED_mart.jpg"
# table_extractor             = te.TableExtractor(path_to_image)
# perspective_corrected_image = table_extractor.execute()
# cv2.imshow("perspective_corrected_image", perspective_corrected_image)


# lines_remover       = tlr.TableLinesRemover(perspective_corrected_image)
# image_without_lines = lines_remover.execute()
# cv2.imshow("image_without_lines", image_without_lines)

# ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
# ocr_tool.execute()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2
# import os

# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2
# import os

# # Path to image
# path_to_image = r"D:\Pingaaksh_System\Main_Ocr\image\LED_mart.jpg"

# # Check if the image exists
# if not os.path.exists(path_to_image):
#     print(f"Image file not found at {path_to_image}")
# else:
#     # Initialize TableExtractor and run the process
#     try:
#         table_extractor = te.TableExtractor(path_to_image)
#         perspective_corrected_image = table_extractor.execute()
#         cv2.imshow("Perspective Corrected Image", perspective_corrected_image)

#         # Remove lines from the image
#         lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
#         image_without_lines = lines_remover.execute()
#         cv2.imshow("Image Without Lines", image_without_lines)

#         # Run OCR to extract table data
#         ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
#         ocr_tool.execute()

#     except Exception as e:
#         print(f"Error occurred: {e}")

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#--------------------------------------------------------------------------------
# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2
# import os

# # Path to image
# path_to_image = "./image/Led_mart9.jpg"

# if not os.path.exists(path_to_image):
#     print(f"Image file not found at {path_to_image}")
# else:
#     try:
#         # Table extraction
#         table_extractor             = te.TableExtractor(path_to_image)
#         perspective_corrected_image = table_extractor.execute()
#         cv2.imshow("Perspective Corrected Image", perspective_corrected_image)

#         # Line removal
#         lines_remover       = tlr.TableLinesRemover(perspective_corrected_image)
#         image_without_lines = lines_remover.execute()
#         cv2.imshow("Image Without Lines", image_without_lines)

#         # OCR to extract table data
#         ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
#         ocr_tool.execute()

#     except Exception as e:
#         print(f"Error occurred: {e}")

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2
# import os

# # Path to image
# path_to_image = "./image/Led_mart9.jpg"

# if not os.path.exists(path_to_image):
#     print(f"Image file not found at {path_to_image}")
# else:
#     try:
#         # Step 1: Table extraction
#         table_extractor = te.TableExtractor(path_to_image)
#         perspective_corrected_image = table_extractor.execute()
#         cv2.imshow("Perspective Corrected Image", perspective_corrected_image)

#         # Step 2: Line removal
#         lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
#         image_without_lines = lines_remover.execute()
#         cv2.imshow("Image Without Lines", image_without_lines)

#         # Step 3: OCR to extract table data
#         ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
#         ocr_tool.execute()

#     except Exception as e:
#         print(f"Error occurred: {e}")

#     # Wait for a key press and close windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#==========================================================================
#Main.py
import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import MainTableExtractor as mte
import cv2
import os
import SignatureRemover as ssp

# Path to image
path_to_image = "/Users/parshvapatel/Desktop/Docsmart/RD/Radhe-Invoices/NeelKanth.jpeg"


if not os.path.exists(path_to_image):
    print(f"Image file not found at {path_to_image}")
else:
    try:
        # Table extraction (Perspective Correction)
        table_extractor = te.TableExtractor(path_to_image)
        perspective_corrected_image = table_extractor.execute()
        #cv2.imshow("Perspective Corrected Image", perspective_corrected_image)

        # Main table extraction (header detection, SI No matching)
        main_table_extractor = mte.MainTableExtractor(perspective_corrected_image)
        main_table_image = main_table_extractor.execute()
        
        
        if main_table_image is None:
            raise Exception("Main table not found in the image.")
        signature_removed=ssp.SignatureRemover(main_table_image)
        removed_signatures_and_arcs=signature_removed.remove_blue_signature()

        # Now remove lines from the cropped main table image
        lines_remover = tlr.TableLinesRemover(removed_signatures_and_arcs)  # Use cropped main table
        image_without_lines = lines_remover.execute()
        #cv2.imshow("Image Without Lines", image_without_lines)

        # OCR to extract table data
        ocr_tool = ottt.OcrToTableTool(image_without_lines, main_table_image)
        ocr_tool.execute()

        # Print extracted table data
        print("\nExtracted Table Data:")
        for row in ocr_tool.table:
            print(", ".join(row))

    except Exception as e:
         print(f"Error occurred: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
