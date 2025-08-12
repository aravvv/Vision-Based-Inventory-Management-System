'''import telebot
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os

# Replace with your actual bot token
API_TOKEN = '8107354407:AAEgwzx-5Hdq_tFF07L8nyH8soIhEz1Lp5E'

# Initialize the bot with the token
bot = telebot.TeleBot(API_TOKEN)

# Initialize a state variable to track whether the bot is waiting for a photo
waiting_for_photo = {}

# Function to handle the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Please type 'incoming' and I will ask you to send a photo to process.")

# Function to handle the 'incoming' command
@bot.message_handler(func=lambda message: message.text.lower() == 'incoming')
def ask_for_photo(message):
    bot.reply_to(message, "Please send me a photo now.")
    waiting_for_photo[message.chat.id] = True  # Set the state to 'waiting for photo'
    print(f"User {message.chat.id} is now waiting for a photo.")  # Debug print

# Function to handle image sent by the user
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print(f"Received photo from user {message.chat.id}")  # Debug print
    
    # Check if the bot is waiting for a photo from the user
    if message.chat.id in waiting_for_photo and waiting_for_photo[message.chat.id]:
        # Once the bot receives the image, stop waiting
        waiting_for_photo[message.chat.id] = False
        print(f"User {message.chat.id} sent a photo.")  # Debug print
        
        # Get the file ID of the sent image
        file_id = message.photo[-1].file_id

        # Get the file information (this includes the file path)
        file_info = bot.get_file(file_id)

        # Debug print to check if file_info is valid
        print(f"File info: {file_info}")
        
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
            print("Image opened successfully.")  # Debug print
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

            # Send the cropped image back to the user
            with open(output_path, 'rb') as cropped_image:
                bot.send_photo(message.chat.id, cropped_image)

            os.remove(output_path)
            print(f"Deleted cropped image from {output_path}")

    else:
        print(f"User {message.chat.id} is not in the waiting state.")  # Debug print
        bot.reply_to(message, "Please type 'send photo' to start the process.")

# Start the bot
bot.polling()
'''

import telebot
import cv2
import numpy as np
import easyocr
import requests
from io import BytesIO
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual bot token
API_TOKEN = '8107354407:AAEgwzx-5Hdq_tFF07L8nyH8soIhEz1Lp5E'

# Initialize the bot with the token
bot = telebot.TeleBot(API_TOKEN)

# Initialize a state variable to track whether the bot is waiting for a photo
waiting_for_photo = {}

# Initialize easyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you want to use CPU

# Function to handle the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Please type 'incoming' and I will ask you to send a photo to process.")

# Function to handle the 'incoming' command
@bot.message_handler(func=lambda message: message.text.lower() == 'incoming')
def ask_for_photo(message):
    bot.reply_to(message, "Please send me a photo now.")
    waiting_for_photo[message.chat.id] = True  # Set the state to 'waiting for photo'
    print(f"User {message.chat.id} is now waiting for a photo.")  # Debug print

# Function to handle image sent by the user
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print(f"Received photo from user {message.chat.id}")  # Debug print
    
    # Check if the bot is waiting for a photo from the user
    if message.chat.id in waiting_for_photo and waiting_for_photo[message.chat.id]:
        # Once the bot receives the image, stop waiting
        waiting_for_photo[message.chat.id] = False
        print(f"User {message.chat.id} sent a photo.")  # Debug print
        
        # Get the file ID of the sent image
        file_id = message.photo[-1].file_id

        # Get the file information (this includes the file path)
        file_info = bot.get_file(file_id)

        # Debug print to check if file_info is valid
        print(f"File info: {file_info}")
        
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
            print("Image opened successfully.")  # Debug print
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

            # Perform OCR to extract text from the cropped image using easyOCR
            result = reader.readtext(cropped)
            full_text = ""

            # Loop through each detected text box and append the detected text to the full_text
            for detection in result:
                detected_text = detection[1]  # Extracting the text part of the detection result
                full_text += detected_text + "\n"  # Adding each line of text

            print("Extracted Text:")
            print(full_text)  # Debug print

            # Process extracted text (e.g., split it into lines)
            data = []
            for line in full_text.split("\n"):
                if line.strip():  # Only add non-empty lines
                    data.append(line.strip())

            # Save extracted data to an Excel file
            excel_output_path = "output_images/extracted_data.xlsx"
            df = pd.DataFrame(data, columns=["Extracted Data"])
            df.to_excel(excel_output_path, index=False)
            print(f"Extracted data saved to {excel_output_path}")  # Debug print

            # Send the cropped image back to the user
            with open(output_path, 'rb') as cropped_image:
                bot.send_photo(message.chat.id, cropped_image)

            # Send the extracted data as a file
            with open(excel_output_path, 'rb') as excel_file:
                bot.send_document(message.chat.id, excel_file)

            # Delete the files after sending them to the user
            os.remove(output_path)
            os.remove(excel_output_path)
            print(f"Deleted files: {output_path}, {excel_output_path}")  # Debug print

    else:
        print(f"User {message.chat.id} is not in the waiting state.")  # Debug print
        bot.reply_to(message, "Please type 'send photo' to start the process.")

# Start the bot
bot.polling()
