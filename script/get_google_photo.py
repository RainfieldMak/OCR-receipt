from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
from googleapiclient.discovery import build,Resource
import pandas as pd
from pandas import DataFrame
import requests


#BUG  cannot download photo with original resolution ( unsolved), https://issuetracker.google.com/issues/112096115


SCOPES=['https://www.googleapis.com/auth/photoslibrary.readonly']


#https://developers.google.com/people/quickstart/python
def get_client_secret(file_name) -> Credentials:

    creds=None
    path = os.path.join("..","env","token.json")
    if os.path.exists(path):
        creds=Credentials.from_authorized_user_file(path, SCOPES)
    
     # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join("..","env",file_name), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(path, 'w') as token:
            token.write(creds.to_json())

    return creds

#need to include static_discovery=false, y ??
def build_service( creds) -> Resource :
    API_NAME = 'photoslibrary'
    API_VERSION = 'v1'
    service= build(API_NAME,API_VERSION,credentials=creds,static_discovery=False)
    return service


#return a dict of album which contain the album name and album id 
def get_album_list(service) -> dict:

    re= service.albums().list().execute()

    if "albums" not in re:
        return album_list

    albums_data = re["albums"]
    album_list = {item["title"]:item["id"] for item in albums_data}
    return album_list


#(https://github.com/googleapis/google-api-python-client/issues/1764)
def get_album_items_info_list(service, album_id) -> list:

    out=[]
    token= None

    
    while True:

        #data= service.mediaItems().search(albumID=album_id).execute()   (bug)

        media_items = service.mediaItems().search(body=dict(albumId=album_id, pageToken=token)).execute()

        out+= media_items["mediaItems"]

        #next page token not exist , i.e. reach the end of item
        if "nextPageToken" not in media_items:
            break
        else:
            token = media_items["nextPageToken"]

    return out
    

def convert_info_db(media_items_list) -> DataFrame:
    # Create an empty list to store individual DataFrames
    data_frames = []
    
    for item in media_items_list:
        # Extract relevant information from the dictionary
        item_id = item["id"]
        filename = item["filename"]
        creation_time = item["mediaMetadata"]["creationTime"]
        height = item["mediaMetadata"]["height"]
        width = item["mediaMetadata"]["width"]
        base_url = item["baseUrl"]
        
        # Create a DataFrame for the current item
        df_item = pd.DataFrame({"id": [item_id], "filename": [filename], "creation_time": [creation_time],
                                "height": [height], "width": [width], "base_url": [base_url]})
        
        # Append the DataFrame to the list
        data_frames.append(df_item)
    
    # Concatenate all individual DataFrames into a single DataFrame
    df = pd.concat(data_frames, ignore_index=True)
    
    return df


def batch_download_album_photo(df) -> int:

    for index, row in df.iterrows():
        base_url= row["base_url"]
        file_name= row["filename"]

        try:
            response = requests.get(base_url)
            if response.status_code == 200:
            # Get the content of the response and save it to a file
                with open(os.path.join("..","data","raw_image",file_name), "wb") as file:
                    file.write(response.content)
                    print(f" {file_name} downloaded successfully.")
            else:
                print(f"{file_name} fail to download - status code : {response.status_code}.")
                

        except Exception as e:
            print(f"error as {e}")
            return -1
        
    return len(df)


#debug test to see if base_url + =d will download photo with original resolution:
def singular_download(base_url) -> bool:
    base_url+="=w2048-h1024"

    try:
            response = requests.get(base_url)
            if response.status_code == 200:
            # Get the content of the response and save it to a file
                with open(os.path.join("..","data","raw_image","IMG_6522.HEIC"), "wb") as file:
                    file.write(response.content)
                    print(f"  downloaded successfully.")
            else:
                print(f" fail to download - status code : {response.status_code}.")
                

    except Exception as e:
        print(f"error as {e}")
        return False
    
    return True


def get_receipt_photo()->DataFrame:

    client_secret_file="credentials.json"

    creds= get_client_secret(client_secret_file)
    service= build_service(creds)

    try:    
        album_list= get_album_list(service)
          

    except Exception as e:
        print(print(f"Error while getting album list: {e}"))


    if "Receipt" in album_list:
        media_item_list= get_album_items_info_list(service, album_list["Receipt"])
        
    else:
        print("Cannot find receipt album")
        return None
    
    if media_item_list:
        media_df= convert_info_db(media_item_list)
        print(media_df)

    else:
        return None

    if  len(media_df) != 0 and batch_download_album_photo(media_df) != -1 :

        print("Download complete")
        return media_df
    else:
        return None
        

