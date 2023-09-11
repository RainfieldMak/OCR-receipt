from PIL import Image
import os
from pillow_heif import register_heif_opener


#Assume image taken from an iphone
def convert_to_jpg(file_name):

    image_path = os.path.join("..", "data", "raw_image", file_name)
    base_name, file_extention= os.path.splitext(file_name)
    saved_file_name=f"{base_name}.jpg"
    saved_path= os.path.join("..", "data", "jpg", saved_file_name)

    # Open HEIF or HEIC file
    image = Image.open(image_path)
    if image is None:
        return False


    # Convert to JPEG
    image.convert('RGB').save( saved_path)
    return True



def batch_conversion():
    register_heif_opener()

    image_folder_path = os.path.join("..", "data", "raw_image")
    file_list= [photo for photo in os.listdir(image_folder_path) if ".HEIC" in photo]

    for photo in file_list:
        if convert_to_jpg(photo):
            print(photo, " successfully convert to jpg")
        else:
            print(photo, " conversion failed")



def get_file_list():

    image_folder_path = os.path.join("..", "data", "jpg")
    file_list= [photo for photo in os.listdir(image_folder_path) if ".jpg" in photo]
    return file_list


