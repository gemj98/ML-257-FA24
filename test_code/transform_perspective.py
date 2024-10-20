import cv2 
import numpy as np 
from pathlib import Path

# Get the current directory and create the relative path to the image
current_dir = Path(__file__).parent
img_path = current_dir / 'images' / 'abc.jpg'

while True:
     
    # ret, frame = cap.read()
    frame = cv2.imread(img_path)
 
    # Locate points of the documents
    # or object which you want to transform
    height, width, channels = frame.shape
    pts1 = np.float32([[88, 27], [514, 0],
                       [width, height], [88-323, height-20]])
    pts2 = np.float32([[0, 0], [width, 0],
                       [width, height], [0, height]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (width, height))
    
    # The array of points needs to be reshaped for cv2.polylines
    pts1b = pts1.astype(np.int32).reshape((-1, 1, 2))
    # Define the color (BGR) and thickness of the polyline
    color = (0, 255, 0)  # Green color
    thickness = 3
    # Draw the polygon
    frame = cv2.polylines(frame, [pts1b], True, color, thickness)
    
    # Wrap the transformed image
    cv2.imshow('frame', frame) # Initial Capture
    cv2.imshow('frame1', result) # Transformed Capture
 
    # Exit the code by pressing 'Esc'
    if cv2.waitKey(24) == 27:
        break
 
# cap.release()
cv2.destroyAllWindows()