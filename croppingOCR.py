import cv2
import numpy as np

# Load the original image
image_path = "D:\TeleBot\godrej\sampless\sample5.jpg"  # Path to your uploaded image
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv_image)
# Define the pink color range in HSV
lower_pink = np.array([140, 50, 50])  # Adjust these values if needed
upper_pink = np.array([180, 255, 255])

# Create a mask for the pink color
mask = cv2.inRange(hsv_image, lower_pink, upper_pink)
cv2.imshow("mask", mask)
# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("No pink region detected. Adjust the color range or check the image.")
else:
    # Find the largest contour (assuming it's the pink box)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the new cropping boundaries to exclude the left column
    left_column_width = int(0.30 * w)  # Adjust the percentage as needed
    new_x = x + left_column_width  # Shift the left boundary to exclude the column
    new_w = w - left_column_width  # Adjust the width to exclude the left column

    # Ensure the dimensions are within the image bounds
    new_x = max(0, new_x)
    cropped = image[y:y+h, new_x:x+w]

    output_path = "/mnt/data/cropped_adjusted_output.jpg"
    #cv2.imwrite(output_path, cropped)
    #print(f"Cropped image saved at {output_path}")

    # Optional: Display the cropped image
    cv2.imshow("Cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    