import cv2

mask_image_path = r"data\mask_1920_1080.png"
video_path = r"data\parking_1920_1080.mp4"

video_capture = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# cv2.imshow('Window Name', mask)
# cv2.waitKey(0)  # Wait for a key press
# cv2.destroyAllWindows()  # Close all OpenCV windows

def draw_bounding_boxes(frame, mask, step_by_step=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if step_by_step:
        cv2.imshow("Image", gray)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Exit if 'q' is pressed
            return None
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    if step_by_step:
        cv2.imshow("Image", segmented)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Exit if 'q' is pressed
            return None
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def show_image(title, img, neww):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, img)
    h, w = img.shape[0:2]
    newh = int(neww*(h/w))
    cv2.resizeWindow(title, neww, newh)  # Resize to 640x480, for example
    
# New code to process video
if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    while video_capture.isOpened():
        print("video is open")
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to read frame from video or end of video reached.")
            break
    
        # Process the frame
        processed_frame = draw_bounding_boxes(frame, mask, True)
        
        # If the user pressed 'q' during the step-by-step process, exit the loop
        if processed_frame is None:
            break
        
        # Display the frame in the resizable window
        show_image('Image', processed_frame, 1200)
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Exit if 'q' is pressed
            break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()