import cv2
import os
import sys

def gather_images(label_name, num_samples):
    # Create directories for the images
    IMG_SAVE_PATH = 'image_data'
    IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

    os.makedirs(IMG_SAVE_PATH, exist_ok=True)
    os.makedirs(IMG_CLASS_PATH, exist_ok=True)

    # Initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # Define the ROI rectangle
    roi_top_left = (100, 100)
    roi_bottom_right = (500, 500)
    roi_color = (255, 255, 255)
    roi_thickness = 2

    # Define the font for displaying status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 255)
    font_thickness = 2

    # Initialize state variables
    collecting = False
    count = 0

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, roi_color, roi_thickness)

        # Display the status text
        status_text = f"Collecting {count}/{num_samples} samples"
        cv2.putText(frame, status_text, (5, 50), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Display the frame on the screen
        cv2.imshow("Collecting images", frame)

        # Check for user input
        key = cv2.waitKey(1)
        if key == ord('a'):
            collecting = not collecting
        elif key == ord('q'):
            break

        # Collect images if collecting is active
        if collecting:
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            save_path = os.path.join(IMG_CLASS_PATH, f"{count}.jpg")
            cv2.imwrite(save_path, roi)
            count += 1

            # Stop collecting if the desired number of images has been reached
            if count == num_samples:
                collecting = False
                break

    # Release the webcam and destroy the display window
    cap.release()
    cv2.destroyAllWindows()

    # Print a message with the number of images saved
    print(f"\n{count} image(s) saved to {IMG_CLASS_PATH}")
