import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from PIL import Image

# "C:\Unet\saved img\Jungle\IMG\center_2024_03_01_13_46_05_282.png"
image_path = '/mnt/c/Unet/saved img/Jungle/IMG/road_2024_03_01_13_46_05_282.png'
image_bgr = cv2.imread(image_path)
SAVING_PATH = '/mnt/c/Unet/saved mask/'

# "C:\Unet\saved img\Jungle\driving_log.csv"
# Pfad zur CSV-Datei
csv_datei_pfad = "/mnt/c/Unet/saved img/Jungle/driving_log.csv"

# Mit einem for loop durch jedes Bildpaar iterieren
with open(csv_datei_pfad, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        image1_path = row[0].strip().replace("\\", "/").replace("C:/Coding",
                                                                "/mnt/c/Unet")  # Remove any leading/trailing whitespace
        image2_path = row[3].strip().replace("\\", "/").replace("C:/Coding",
                                                                "/mnt/c/Unet")  # Remove any leading/trailing whitespace

        originalIMG = Image.open(image1_path)
        maskIMG = Image.open(image2_path)
        #originalIMG = originalIMG.resize((640,320))
        #maskIMG = maskIMG.resize((640, 320))
        print(originalIMG.size)
        width, height = originalIMG.size
        resultIMG = Image.new(originalIMG.mode, (width, height))
        #background color = (0, 255, 1)
        for y in range(height):
            for x in range(width):
                original_pixel_value = originalIMG.getpixel((x, y))
                mask_pixel_value = maskIMG.getpixel((x, y))
                if mask_pixel_value == (0, 255, 1): resultIMG.putpixel((x, y), original_pixel_value)
                if mask_pixel_value == original_pixel_value: resultIMG.putpixel((x, y), mask_pixel_value)
                else:resultIMG.putpixel((x, y), (0, 255, 1))


                '''
                if original_pixel_value == (57, 63, 53):
                    resultIMG.putpixel((x, y), original_pixel_value)
                print(original_pixel_value)
                '''


        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(originalIMG)
        axes[0].axis('off')
        axes[1].imshow(maskIMG)
        axes[1].axis('off')
        axes[2].imshow(resultIMG)
        axes[2].axis('off')
        plt.show()
        exit()
