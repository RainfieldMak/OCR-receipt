from boundingbox import *
from get_text import get_paid_amount,back_up_plan
from scan_image_preprocessing import *
from scan_image_preprocessing import denoising, save_scann_file
import pytesseract
from PIL import Image
from file_conversion import batch_conversion, get_file_list
import os


# for getting the total amount in one receipt
def singular_testing(file_name)->None:

    tessart_path= os.path.join('C:\Program Files\Tesseract-OCR', 'tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = tessart_path

    image_path = os.path.join("..", "data", "jpg", file_name)

    scanned=get_receipt(image_path)

    
    denoise_scanned_image= denoising(scanned)
    save_scann_file(file_name, Image.fromarray(denoise_scanned_image))
    denoise_scanned_image.show()

    scann_text=pytesseract.image_to_string(denoise_scanned_image)
    #print(scann_text)


    total = get_paid_amount(str(scann_text))
    if total is None:
        total= back_up_plan(str(scann_text))

    if total is None:
        print(f"[{file_name}] failed")
    else:
        print(f"[{file_name}]Total amout is {total}")



# for batch testing to get the total amount of a large amount of receipt
def batch_testing(file_list):


    fail_list=[]
    success_list=[]
    tessart_path= os.path.join('C:\Program Files\Tesseract-OCR', 'tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = tessart_path

    for file in file_list:
        image_path = os.path.join("..", "data", "jpg", file)

        scanned=get_receipt(image_path)
       # print(type(scanned))

        denoise_scanned_image= denoising(scanned)
        save_scann_file(file, Image.fromarray(denoise_scanned_image))
        

        scann_text=pytesseract.image_to_string(denoise_scanned_image)

      
        #print(scann_text)


        total = get_paid_amount(str(scann_text))
        if total is None:
            total= back_up_plan(str(scann_text))

        if total is None:
            #print(f"[{file}] failed")
            save_failed_image(Image.fromarray(denoise_scanned_image) , file)
            fail_list.append(file)
        else:
            sus=f"[{file}]Total amout is {total}"
            print(sus)
            success_list.append(sus)

        #print(f"[{file}] done")
        print("************************************************************************")


    failed_path= os.path.join("..","data","failed","failed.txt") 
    fail_list_str = '\n'.join(fail_list)
    with open(failed_path,"w") as file:
        file.write(fail_list_str)
        
    return success_list


if __name__ == "__main__":

    try:
        batch_conversion()
    except Exception as e:
        print(f"Error while converting file {e}")

    try:    
        sus_list=batch_testing(get_file_list())
    except Exception as e:
        print(f"Error getting receipt total {e}")