import cv2
import numpy as np

# Load the image
image_path = 'ball.png'
image = cv2.imread(image_path)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Ball colors range definition
lower_mask1 = np.array([0, 100, 10])
upper_mask1 = np.array([10, 255, 255])

lower_mask2 = np.array([160,100,20])
upper_mask2 = np.array([180,255,255])

# Threshold the HSV image to get only ball colors
mask1 = cv2.inRange(hsv, lower_mask1, upper_mask1)
mask2 = cv2.inRange(hsv, lower_mask2, upper_mask2)

# Combine the 2 masks
mask_combined = cv2.bitwise_or(mask1, mask2)

# Perform morphological operations to remove small noise
kernel = np.ones((5, 5), np.uint8)
mask_combined1 = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
mask_combined1 = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

# Find contours and the center of the ball
contours, _ = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    c = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    if radius > 0:
        cv2.circle(image, center, int(radius), (0, 255, 255), 2)
        cv2.circle(image, center, 5, (0, 0, 255), -1)

# Display the result image
        
cv2.imshow('Mask', mask_combined)
cv2.imshow('Processed Image', image)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Ensure all windows will be closed
