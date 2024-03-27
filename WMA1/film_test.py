import cv2
import numpy as np

# Open the video
video = cv2.VideoCapture()
video.open('movingball.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Prepare the video writer
result = cv2.VideoWriter(
    'result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size
)

counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break
    print(f'Frame {counter} of {total_frames}')

    # Convert to HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)

    # Define range of orange color in HSV
    lower_mask1 = np.array([0, 100, 10])
    upper_mask1 = np.array([10, 255, 255])
    lower_mask2 = np.array([160, 100, 20])
    upper_mask2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask1 = cv2.inRange(hsv, lower_mask1, upper_mask1)
    mask2 = cv2.inRange(hsv, lower_mask2, upper_mask2)

    # Combine the 2 masks
    mask_combined = cv2.bitwise_or(mask1, mask2)

    # Perform morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    # Find contours and the center of the ball
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        if radius > 0:
            cv2.circle(frame_rgb, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame_rgb, center, 5, (0, 0, 255), -1)

    # Write the frame to the result video
    result.write(frame_rgb)
    counter += 1

# Release the video objects
video.release()
result.release()
