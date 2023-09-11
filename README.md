# OCR-receipt

# Author: 
    Rainfield Mak 

# Description

    1.A scanned version (croped, grey scale, .jpg file) version of the original image will be store in ./scanned

    2.Using OCR technique to obtain the total payable amount on a receipt image (taken from Iphone, .HEIC file).





# How to run 


   ## Required File Structure .
        .
        |-- data
        |   |-- canny_image
        |   |-- failed
        |   |-- jpg
        |   |-- raw_image
        |   `-- scanned
        |-- env
        `-- script


   ## To run
        python3 main.py 


# Usage 
    1. Default use is for batch conversion for image stored in ./raw folder.(Singular file conversion is also support for debug purpose)

    2. Script support download image from google photo using REST API (!! downloaded will be downsampled, not recommend to use since will greatly affect OCR accuracy)


# Example 

![image](https://github.com/RainfieldMak/OCR-receipt/assets/130533588/20fe893a-73e6-46ef-b9ad-b64b1e8f525d)

