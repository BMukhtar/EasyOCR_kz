import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os

import sys
sys.path.append('./')

import easyocr

# add ../easyocr package to sys


print(easyocr.__file__)

reader = easyocr.Reader(
    ['en'],
    gpu=True,
    recog_network='best_norm_ED',
    user_network_directory='./models',
    model_storage_directory='./models',
)  # this needs to run only once to load the model into memory

images = ['kitap7.png']
# images = ['book_easy_1.jpg', 'kz_book_simple.jpeg', 'kz_blur.jpg', 'kz_book_complex.jpg', '20230629_160049.jpg']

for image_name in tqdm(images):
    # Read image as numpy array
    image = cv2.imread("./examples/" + image_name)

    # Rotate the image by 270 degrees
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Convert the image from BGR to RGB (because OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sourceImage = image.copy()
    results = reader.readtext(
        image=image, 
        batch_size=64,
        width_ths = 0
        )

    # Load custom font
    font_path = "./test_model/Ubuntu-Regular.ttf"
    

    # Display the results
    for (bbox, text, prob) in results:
        # Get the bounding box coordinates
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        # height
        h = bottom_right[1] - top_left[1]
        # width
        w = bottom_right[0] - top_left[0]
        # define font size based on height
        font_size = max(int(h / 3), 10)
        font = ImageFont.truetype(font_path, font_size)

        # Draw the bounding box on the image
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Convert the OpenCV image to a PIL image, draw the text, then convert back to an OpenCV image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        draw.text((top_left[0]+int(w/3), top_left[1]-int(font_size/2)), text, font=font, fill=(0, 0, 255))
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite('./test_model/output/' + image_name, image)

    wholeText = reader.readtext(image = sourceImage, batch_size=64, paragraph=True, y_ths=0, width_ths = 0)
    # write to file
    with open('./test_model/output/' + image_name + '.txt', 'w') as f:
        for text in wholeText:
            f.write(text[1] + '\n')  
        

