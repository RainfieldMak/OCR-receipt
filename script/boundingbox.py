import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local
from PIL import Image


def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plot_gray(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(image, cmap='Greys_r')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

def detect_receipt_edges(image):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    return edges

def extract_receipt_contours(edges):
    # Find contours in the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def get_bounding_rectangle_points(contours):
    # Merge all the points into a single array
    points = np.vstack(contours)

    # Find the bounding rectangle that encompasses all the points
    x, y, w, h = cv2.boundingRect(points)

    # Calculate the four corner points of the rectangle
    top_left = [x, y]
    top_right = [x + w, y]
    bottom_left = [x, y + h]
    bottom_right = [x + w, y + h]

    # Format the points as requested
    rectangle_points = np.array([top_left, top_right, bottom_left, bottom_right])

    return rectangle_points



def approximate_open_polygon_to_rectangle(contour):
    
    # Extract the x and y coordinates of all points in the contour
    x_coordinates = [point[0][0] for point in contour]
    y_coordinates = [point[0][1] for point in contour]

    # Find the lower-left and upper-right corners
    lower_left = (min(x_coordinates), min(y_coordinates))
    upper_right = (max(x_coordinates), max(y_coordinates))

    # Create a rectangle with four points
    rect = np.array([
        [lower_left[0], lower_left[1]],
        [upper_right[0], lower_left[1]],
        [upper_right[0], upper_right[1]],
        [lower_left[0], upper_right[1]]
    ], dtype=np.int32)

    return np.array([rect], dtype=np.int32)  # Return the rectangle as a list of points


#straight up max (area of all contour) would not work since some maybe a open instead of closed polygon
def get_receipt_contour(contours):

    rect_list=[]
    for contour in contours:
        if is_open_polygon(contour):
            rect_list.append (approximate_open_polygon_to_rectangle(contour))
        else:
            rect_list.append(approximate_contour(contour))
    

    
    #assume rect with largest area would be the area covered by receipt
    return max(rect_list, key=cv2.contourArea)


# Approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    epsilon = 5 * peri  # Adjust this epsilon value as needed
    
    
    return cv2.approxPolyDP(contour, epsilon, True)


#check if the contour resemble a  open polygon
def is_open_polygon(contour):
    # Get the starting and ending points of the contour
    start_point = tuple(contour[0][0])
    end_point = tuple(contour[-1][0])

    # Check if the starting and ending points are different
    return start_point != end_point





def contour_to_rect(contour,resize_ratio):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255



def save_canny_image(image, file_name):
    canny_image =  Image.fromarray(image)
    base_name, file_extention= os.path.splitext(file_name)
    saved_file_name=f"{base_name}.jpg"
    saved_path= os.path.join("..", "data", "canny_image", saved_file_name)
    canny_image.save(saved_path)



def draw_contours_with_colors(image, contours_list):
    # Make a copy of the input image to avoid modifying the original
    result_image = image.copy()

    # Define a list of colors in a rainbow-like sequence
    colors = [
        (255, 0, 0),    # Red
        (255, 165, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 128, 0),    # Green
        (0, 0, 255),    # Blue
        (75, 0, 130),   # Indigo
        (128, 0, 128),  # Violet
        (0, 0, 0),      # Black
        (255, 255, 255),# White
        (0, 100, 0),    # Dark Green
    ]

    # Initialize a counter for cycling through colors
    color_index = 0

    for i, contour in enumerate(contours_list):


    
        # Get the current color
        color = colors[color_index]

        area = cv2.contourArea(contour)

        # Draw the contour on the result image
        cv2.drawContours(result_image, [contour], -1, color, thickness=cv2.FILLED)

        # Increment the color index, cycling through the colors
        color_index = (color_index + 1) % len(colors)

    return result_image



def calculate_contour_areas(contours_list):
    contour_areas = []
    
    for contour in contours_list:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        contour_areas.append(area)
    
    return contour_areas



def get_receipt(image_path):

    img = cv2.imread(image_path)
   
    # Resize the image 
    resize_ratio = 500 / img.shape[0]
    image = opencv_resize(img, resize_ratio)


    # Preprocess the image (convert to grayscale and apply Gaussian blur)
    preprocessed_image = preprocess_image(image)
    


    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(preprocessed_image, rectKernel)
 

    # Detect edges
    edges = detect_receipt_edges(dilated)

    
    save_canny_image(edges, os.path.basename(image_path))

    # Extract contours
    contours = extract_receipt_contours(edges)


    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]


# debug **************************************************
#     result_image = draw_contours_with_colors(   image.copy(),  largest_contours)
#    # cv2.imshow('Contours', result_image)
    

        
#     contours_areas = calculate_contour_areas(largest_contours)

#     color_names = [
#         "Red",
#         "Orange",
#         "Yellow",
#         "Green",
#         "Blue",
#         "Indigo",
#         "Violet",
#         "Black",
#         "White",
#         "Dark Green",
#     ]


#     # Print or use the contour areas as needed
#    # for i, area, color in zip(range(10),contours_areas, color_names):
#     #    print(f"{color}Contour {i+1} Area:", area)


#**************************************************

    receipt_contour = get_receipt_contour(largest_contours)
    
    final= cv2.drawContours(image.copy(),[receipt_contour],-1, (0, 255, 0), 2)

    scanned = wrap_perspective(img.copy(), contour_to_rect(receipt_contour,resize_ratio))

    result = bw_scanner(scanned)

    return result













