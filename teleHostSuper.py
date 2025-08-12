import telebot
import easyocr
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import os
import re

# Replace with your actual bot token
API_TOKEN = '8107354407:AAEgwzx-5Hdq_tFF07L8nyH8soIhEz1Lp5E'

# Initialize the bot with the token
bot = telebot.TeleBot(API_TOKEN)

# Initialize easyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Function to handle the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Please type 'incoming' and I will ask you to send a photo to process.")

# Function to handle the 'incoming' command
@bot.message_handler(func=lambda message: message.text.lower() == 'incoming')
def ask_for_photo(message):
    bot.reply_to(message, "Please send me a photo now.")

# Function to handle image sent by the user
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print(f"Received photo from user {message.chat.id}")  # Debug print
    
    # Get the file ID of the sent image
    file_id = message.photo[-1].file_id

    # Get the file information (this includes the file path)
    file_info = bot.get_file(file_id)

    # Download the image
    file_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}"
    response = requests.get(file_url)

    if response.status_code != 200:
        print(f"Failed to download image, status code: {response.status_code}")
        bot.reply_to(message, "There was an issue downloading the image.")
        return

    # Open the image using PIL (Python Imaging Library)
    try:
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error opening image: {e}")
        bot.reply_to(message, "There was an error processing the image.")
        return

    # Convert the image to an OpenCV-compatible format (from PIL to numpy array)
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Define the pink color range in HSV
    lower_pink = np.array([140, 50, 50])  # Adjust these values if needed
    upper_pink = np.array([180, 255, 255])

    # Create a mask for the pink color
    mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No pink region detected.")
        bot.reply_to(message, "No pink region detected. Please try again with a different image.")
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
        cropped = img_cv[y:y+h, new_x:x+w]

        # Create output folder if it doesn't exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")

        # Save the cropped image to a file
        output_path = "output_images/cropped_adjusted_output.jpg"
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved at {output_path}")  # Debug print

        # Perform OCR to extract text from the cropped image using EasyOCR
        result = reader.readtext(cropped)
        full_text = ""

        # Loop through each detected text box and append the detected text to the full_text
        for detection in result:
            detected_text = detection[1]  # Extracting the text part of the detection result
            full_text += detected_text + "\n"  # Adding each line of text

        print("Extracted Text:")
        print(full_text)  # Debug print

        # Look for the 'PRODUCT CODE' section and crop accordingly
        product_code_block = None
        product_value_block = None

        # Check the OCR result for 'PRODUCT CODE' and get the bounding box
        for detection in result:
            detected_text = detection[1]
            if "PRODUCT CODE" in detected_text:
                product_code_block = detection[0]  # This gives us the bounding box

                # To crop the value next to it, we will use a small heuristic
                # Search for the next text box which should be the product code value
                idx = result.index(detection)  # Get index of the product code
                if idx + 1 < len(result):
                    product_value_block = result[idx + 1][0]  # Get the bounding box of the next text

        if product_code_block and product_value_block:
            # Get the bounding box coordinates for both the label and value
            top_left = tuple(product_code_block[0])  # Coordinates for the top-left corner of the product code label
            bottom_right = tuple(product_value_block[2])  # Coordinates for the bottom-right corner of the value box

            # Crop the image to include both the label and the value
            cropped_product_code_value = cropped[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            scale_factor = 2  # Experiment with different scaling factors
            new_width = int(cropped_product_code_value.shape[1] * scale_factor)
            new_height = int(cropped_product_code_value.shape[0] * scale_factor)
            scaled_cropped_image = cv2.resize(cropped_product_code_value, (new_width, new_height))
            # Save this cropped image for the product code and its value
            product_code_value_output_path = "output_images/scaled_cropped_image.jpg"
            cv2.imwrite(product_code_value_output_path, scaled_cropped_image)

            # Perform OCR again on the cropped region (product code and value section)
            result_product_code_value = reader.readtext(scaled_cropped_image)
            product_code_value_text = ""
            for detection in result_product_code_value:
                detected_text = detection[1]
                product_code_value_text += detected_text + " "

            print(f"Extracted Product Code and Value Text: {product_code_value_text}")  # Debug print

            # Send the cropped product code and value image back to the user
            with open(product_code_value_output_path, 'rb') as product_code_value_image:
                bot.send_photo(message.chat.id, product_code_value_image)

            # Send the extracted product code and value text back
            bot.reply_to(message, f"Extracted Product Code and Value: {product_code_value_text}")

            # Delete the cropped product code and value image after sending it
            os.remove(product_code_value_output_path)
            print(f"Deleted file: {product_code_value_output_path}")  # Debug print

        else:
            print("Product Code or Value block not found.")  # Debug print
            bot.reply_to(message, "Couldn't find the Product Code and Value in the image.")

# Start the bot
bot.polling()
