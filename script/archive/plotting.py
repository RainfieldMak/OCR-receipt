import cv2
import numpy as np

# List of contours
contours = [np.array([[[42, 419]], [[48, 425]], [[51, 446]], [[41, 450]], [[40, 459]], [[41, 451]], [[51, 447]], [[48, 424]]])]

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

# Print the four corner points
print(rectangle_points)
