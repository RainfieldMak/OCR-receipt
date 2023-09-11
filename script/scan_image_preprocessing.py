from PIL import Image
import cv2
import os


def save_scann_file(file_name, image):

    try:
        base_name, file_extention= os.path.splitext(file_name)
        saved_file_name=f"{base_name}_scann.jpg"
        saved_path= os.path.join("..", "data", "scanned", saved_file_name)    
        image.save(saved_path)

    except Exception as e:
        print(f"Error while saving the scanned file: {e}")


def save_failed_image(image, file_name):
    base_name, file_extention= os.path.splitext(file_name)
    saved_file_name=f"{base_name}_scan_fail.jpg"
    saved_path= os.path.join("..", "data", "failed", saved_file_name)
    image.save(saved_path)


def denoising(image):

    # Apply denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=40, searchWindowSize=21, templateWindowSize=7)

    return denoised_image



