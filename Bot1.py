import easyocr
import cv2
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

reader = easyocr.Reader(['en'], gpu=True)


image_path = 'D:\TeleBot\godrej\sampless\sample5_cropped.jpg'
image = cv2.imread(image_path)

result = reader.readtext(image)
#print(result)

full_text = ""

# Loop through each detected text box and append the detected text to the full_text
for detection in result:
    detected_text = detection[1]  # Extracting the text part of the detection result
    full_text += detected_text + "\n"  # Adding each line of text

# Print the full extracted text
print("Extracted Text:")
print(full_text)

# Optionally, you can display the image with bounding boxes around the detected text
for detection in result:
    top_left = tuple(detection[0][0])  # Coordinates for the top-left corner
    bottom_right = tuple(detection[0][2])  # Coordinates for the bottom-right corner
    image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Draw bounding box

# Display the image with bounding boxes (if required)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert image to RGB for display in matplotlib
plt.axis('off')  # Hide axes
plt.show()
