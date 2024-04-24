import matplotlib.pyplot as plt
import csv
from PIL import Image
import math
import os

# Pfad zur CSV-Datei
csv_datei_pfad = "/mnt/c/Unet/saved img/Moun/driving_log.csv"
SAVING_PATH = '/mnt/c/Unet/saved mask moun/'


def color_difference(pixel1, pixel2):
    r1, g1, b1 = pixel1
    r2, g2, b2 = pixel2
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def is_color_difference_greater(mask_pixel_value, original_pixel_value, threshold=170):
    return color_difference(mask_pixel_value, original_pixel_value) > threshold


with open(csv_datei_pfad, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        image1_path = row[0].strip().replace("\\", "/").replace("C:/Coding",
                                                                "/mnt/c/Unet")  # Remove any leading/trailing whitespace
        image2_path = row[3].strip().replace("\\", "/").replace("C:/Coding",
                                                                "/mnt/c/Unet")  # Remove any leading/trailing whitespace

        #image1_path = '/mnt/c/Unet/saved img/Jungle/IMG/center_2024_03_01_13_46_16_128.png'
        #image2_path =  '/mnt/c/Unet/saved img/Jungle/IMG/road_2024_03_01_13_46_16_128.png'

        originalIMG = Image.open(image1_path)
        maskIMG = Image.open(image2_path)


        width, height = originalIMG.size
        resultIMG = Image.new(originalIMG.mode, (width, height))
        # background color = (0, 255, 1)

        for y in range(height):
            for x in range(width):
                original_pixel_value = originalIMG.getpixel((x, y))
                mask_pixel_value = maskIMG.getpixel((x, y))

                if mask_pixel_value != (0, 255, 1) and is_color_difference_greater(mask_pixel_value,
                                                                                   original_pixel_value):
                    resultIMG.putpixel((x, y), (0, 255, 1))
                else:
                    resultIMG.putpixel((x, y), mask_pixel_value)
                # turn the road to color red
                if resultIMG.getpixel((x, y)) != (0, 255, 1): resultIMG.putpixel((x, y), (255, 0, 1))

        '''
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(originalIMG)
        axes[0].axis('off')
        axes[1].imshow(maskIMG)
        axes[1].axis('off')
        axes[2].imshow(resultIMG)
        axes[2].axis('off')
        plt.show()
        '''

        filename = os.path.basename(image2_path)
        result_path = os.path.join(SAVING_PATH, filename)
        resultIMG.save(result_path)




print('done')
