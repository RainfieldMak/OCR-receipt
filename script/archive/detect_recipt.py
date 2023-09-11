import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local

class ReceiptDetector:
    def __init__(self):
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    def opencv_resize(self, image, ratio):
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_receipt_edges(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        return edges

    def extract_receipt_contours(self, edges):
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_receipt_contour(self, contours):    
        for c in contours:
            approx = self.approximate_contour(c)
            if len(approx) == 4:
                return approx

    def approximate_contour(self, contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    def contour_to_rect(self, contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # top-left point has the smallest sum
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points:
        # the top-right will have the minimum difference 
        # the bottom-left will have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def wrap_perspective(self, img, rect):
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
            [0, maxHeight - 1]], dtype="float32")
        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        # warp the perspective to grab the screen
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def bw_scanner(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 21, offset=5, method="gaussian")
        return (gray > T).astype("uint8") * 255

    def process_image(self, image_path):
        # Load the image from the provided image path
        img = cv2.imread(image_path)

        # Preprocess the image (convert to grayscale and apply Gaussian blur)
        preprocessed_image = self.preprocess_image(img)

        # Detect edges
        edges = self.detect_receipt_edges(preprocessed_image)

        # Extract contours
        contours = self.extract_receipt_contours(edges)

        # Get the largest contour (receipt contour)
        receipt_contour = self.get_receipt_contour(contours)

        if receipt_contour is not None:
            # Extract the region within the bounding box of the contour
            rect = self.contour_to_rect(receipt_contour)
            extracted_area = self.wrap_perspective(img.copy(), rect)

            # Apply black and white scanning
            result = self.bw_scanner(extracted_area)

            return result

        else:
            return None

def main(image_path):
    detector = ReceiptDetector()
    processed_image = detector.process_image(image_path)
    
    if processed_image is not None:
        print("Image processed successfully.")
        plt.imshow(processed_image, cmap='gray')
        plt.show()
    else:
        print("No receipt contour found.")

if __name__ == "__main__":
    image_path = os.path.join("..", "data", "image", "receipt3.jpg")
    main(image_path)
