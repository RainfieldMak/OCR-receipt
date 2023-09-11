

import os
from boundingbox import get_receipt
import re


def preprocess_text(text):
    # Remove leading/trailing whitespace and replace line breaks with spaces
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    return cleaned_text

def tokenize_text(text):
    return list(map(str.lower,text.split("\n")))

def extract_token(txt_list):
    pattern = r'\btotal\b'
    out=[]
    for item in txt_list:
        if re.search(pattern, item):
            out.append(item)
    return out

def clean_txt(txt):
    return re.sub(r'[^a-zA-Z0-9.]', ' ', txt)


def remove_mix_char_num(txt):
    return re.sub(r'\b[a-zA-Z]+\d+\b', '', txt)

def remove_end_dot(txt):
    pattern = r'(\d+\.\d{2})\b'
    # Use re.sub to replace all matches in the text
    match= re.search(pattern, txt)
    if match:
        return txt.rstrip(".")

def check_for_word(txt):

    substring='sub'
    if txt is not None and substring in txt :
        return None
    else :
        return txt


def find_total_string(txt):

    # Define a regular expression pattern to match the desired pattern
    pattern = r'total\s+\w*\s*\d+\.\d{2}'
    
    # Use re.search to find the pattern in the text
    match = re.search(pattern, txt)
    
    # If a match is found, return True; otherwise, return False
    return bool(match)

def extract_total_string(txt_lst):
    out=[]
    for item in txt_lst:
        if item is not None and find_total_string(item):
            out.append(item)

    return out



def extract_total_amount(lst):

    if lst is None:
        return None
    
    # Define a regular expression pattern to capture the numeric part
    pattern='\d+\.\d{2}'
    temp_=[]
    for item in lst:

        temp=item
        temp=temp.replace("total", "")
        temp=temp.replace(" ","")

        matches= re.findall(pattern,temp)

        if not matches:
            continue
        else:
            temp_.append(float(matches[0]))
    

    if not temp_:
        return None
    else:
        return max(temp_)


def get_paid_amount(txt):
    

    extract_txt= extract_token(tokenize_text(txt))   

    for item in extract_txt:

        temp=clean_txt(item)
        #print("clean text: ", temp)

        temp=remove_mix_char_num(item)
        #print("remove_mix_char_num: ", temp)

        temp=remove_end_dot(item)
        #print("remove_end_dot: ", temp)

        temp=check_for_word(item)
        #print("check_for_word: ", temp)

        item=temp     

    return extract_total_amount(extract_txt)



def detect_dollar_sign(text):

    return [item for item in text if "$" in item]



#assume final one is the total amount
def find_total_dollar(text_lst):
    pattern='\d+\.\d{2}'

    temp_=[]

    for item in text_lst:
        matches=re.findall(pattern, item)
        #print(matches)
        
    
        if not matches:
            continue
        else:
            temp_.append(float(matches[0]))
    
    if not temp_:
        return None
    else:
        return temp_[-1]



def back_up_plan(scann_text):
    text_lst= detect_dollar_sign(tokenize_text(scann_text))

    return find_total_dollar(text_lst)



