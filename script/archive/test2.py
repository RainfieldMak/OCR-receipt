import cv2

def is_open_polygon(contour):
    # Get the starting and ending points of the contour
    start_point = tuple(contour[0][0])
    end_point = tuple(contour[-1][0])

    # Check if the starting and ending points are different
    return start_point != end_point

# Example usage:
# Assuming you have a list of contours (contours_list) obtained from cv2.findContours

import numpy as np

points = np.array([
    [[334, 186]],
    [[335, 187]],
    [[335, 197]],
    [[338, 200]],
    [[348, 200]],
    [[348, 199]],
    [[349, 198]],
    [[349, 190]],
    [[348, 189]],
    [[348, 188]],
    [[347, 187]],
    [[346, 187]],
    [[345, 186]]
], dtype=np.int32)

for i, contour in enumerate(points):
    if is_open_polygon(contour):
        print(f"Contour {i+1} is an open polygon.")
    else:
        print(f"Contour {i+1} is a closed polygon.")
