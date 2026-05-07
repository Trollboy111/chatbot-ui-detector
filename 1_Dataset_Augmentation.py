import cv2
import numpy as np
import random
import os

CHATBOT_NAMES = ["chatgpt", "gemini", "claude"]

os.makedirs("Augmented_Images", exist_ok=True)

def random_scale_window_to_fit_screen(window_image, background_width, background_height):
    min_scale = 0.3
    max_scale = 0.8

    window_height, window_width = window_image.shape[:2]

    max_possible_scale = min(background_width / window_width,background_height / window_height)

    if max_possible_scale < min_scale:
        chosen_scale = max_possible_scale
    else:
        chosen_scale = random.uniform(min_scale,min(max_scale, max_possible_scale))

    resized_width = max(1, int(window_width * chosen_scale))
    resized_height = max(1, int(window_height * chosen_scale))

    return cv2.resize(window_image,(resized_width, resized_height),interpolation=cv2.INTER_AREA)


def random_crop_window_image(window_image):
    crop_chance = 0.3
    if random.random() > crop_chance:
        return window_image

    image_height, image_width = window_image.shape[:2]

    minimum_crop_percentage = 0.6

    cropped_height = random.randint(int(image_height * minimum_crop_percentage), image_height)
    cropped_width = random.randint(int(image_width * minimum_crop_percentage), image_width)

    crop_start_y = random.randint(0, image_height - cropped_height)
    crop_start_x = random.randint(0, image_width - cropped_width)

    return window_image[crop_start_y:crop_start_y + cropped_height,crop_start_x:crop_start_x + cropped_width]

def random_position(background_width, background_height, window_width, window_height):

    maximum_off_screen = 0.5

    min_x_position = 0 - int(window_width * maximum_off_screen)
    max_x_position = background_width - int(window_width * maximum_off_screen)

    min_y_position = 0 - int(window_height * maximum_off_screen)
    max_y_position = background_height - int(window_height * maximum_off_screen)

    window_top_left_x = random.randint(min_x_position, max_x_position)
    window_top_left_y = random.randint(min_y_position, max_y_position)

    onscreen_x1 = max(0, window_top_left_x)
    onscreene_y1 = max(0, window_top_left_y)

    onscreen_x2 = min(background_width, window_top_left_x + window_width)
    onscreen_y2 = min(background_height, window_top_left_y + window_height)

    return (window_top_left_x, window_top_left_y, onscreen_x1, onscreene_y1, onscreen_x2, onscreen_y2)

def sharpen_window_edges_only(image, onscreen_x1, onscreen_y1, onscreen_width, onscreen_height):
    sharpening_kernel = np.array(
        [[0, -1, 0],
         [-1,  5, -1],
         [0, -1, 0]],
        dtype=np.float32
    )

    output_image = image.copy()

    onscreen_image = output_image[onscreen_y1 : onscreen_y1 + onscreen_height, onscreen_x1 : onscreen_x1 + onscreen_width]

    sharpened_region = cv2.filter2D(onscreen_image, -1, sharpening_kernel)

    max_edge_pixels = 3
    edge_width = min(max_edge_pixels, onscreen_width // 2, onscreen_height // 2)

    onscreen_image[:edge_width, :]  = sharpened_region[:edge_width, :]
    onscreen_image[-edge_width:, :] = sharpened_region[-edge_width:, :]
    onscreen_image[:, :edge_width]  = sharpened_region[:, :edge_width]
    onscreen_image[:, -edge_width:] = sharpened_region[:, -edge_width:]

    output_image[onscreen_y1 : onscreen_y1 + onscreen_height, onscreen_x1 : onscreen_x1 + onscreen_width] = onscreen_image

    return output_image


for chatbot_name in CHATBOT_NAMES:
    chatbot_input_dir = os.path.join("chatbot_images", chatbot_name)
    chatbot_output_dir = os.path.join("Augmented_Images", chatbot_name)
    os.makedirs(chatbot_output_dir, exist_ok=True)

    background_image_files = []

    for filename in os.listdir("backgrounds"):
        background_image_files.append(filename)
        
    background_image_files = sorted(background_image_files)

    window_image_files = []

    for filename in os.listdir(chatbot_input_dir):
        window_image_files.append(filename)
        
    window_image_files = sorted(window_image_files)

    for window_image_filename in window_image_files:
        for background_filename in background_image_files:
            background_image = cv2.imread(os.path.join("backgrounds", background_filename))
            window_image = cv2.imread(os.path.join(chatbot_input_dir, window_image_filename))

            background_height, background_width = background_image.shape[:2]

            window_image = random_scale_window_to_fit_screen(window_image,background_width,background_height)
            
            window_image = random_crop_window_image(window_image)

            window_height, window_width = window_image.shape[:2]

            (window_x1, window_y1, onscreen_window_x1, onscreen_window_y1, onscreen_window_x2, onscreen_window_y2) = random_position(background_width, 
                                                                                                                                     background_height, 
                                                                                                                                     window_width,
                                                                                                                                     window_height)

            window_crop_x1 = onscreen_window_x1 - window_x1
            window_crop_y1 = onscreen_window_y1 - window_y1

            onscreen_width  = onscreen_window_x2 - onscreen_window_x1
            onscreen_height = onscreen_window_y2 - onscreen_window_y1

            window_crop_x2 = window_crop_x1 + onscreen_width
            window_crop_y2 = window_crop_y1 + onscreen_height

            background_copy = background_image.copy()

            background_copy[
                onscreen_window_y1:onscreen_window_y2,
                onscreen_window_x1:onscreen_window_x2
            ] = window_image[
                window_crop_y1:window_crop_y2,
                window_crop_x1:window_crop_x2
            ]

            background_copy = sharpen_window_edges_only(background_copy, onscreen_window_x1, onscreen_window_y1, onscreen_width, onscreen_height)

            output_filename = (f"aug_{chatbot_name}_{os.path.splitext(background_filename)[0]}_{os.path.splitext(window_image_filename)[0]}.png")

            cv2.imwrite(os.path.join(chatbot_output_dir, output_filename),background_copy)

print("Augmentation complete.")
