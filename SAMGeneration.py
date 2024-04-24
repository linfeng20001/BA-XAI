import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
#import wget
import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE = "vit_h"
weights_dir = 'C:/Users/Linfe/OneDrive/Desktop/Unet'


#source: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/

#"C:\Users\Linfe\Downloads\sam_vit_h_4b8939.pth"




#/mnt/c/Unet/Dataset
CHECKPOINT_PATH = '/mnt/c/Unet/sam_vit_h_4b8939.pth'
SAVING_PATH = '/mnt/c/Unet/lable/'
FOLDER_PATH = '/mnt/c/Unet/dataset5/track1/normal/IMG'

IMAGE_PATH = 'C:/Users/Linfe/Downloads/dataset5/track1/recovery/IMG/center_2020_07_17_11_19_43_034.jpg'




sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)


for filename in os.listdir(FOLDER_PATH):
    if filename.startswith("center"):  # Adjust extensions as needed
        print(filename)
        # Load image
        image_path = os.path.join(FOLDER_PATH, filename)
        #for this img its the [1] the segment of road
        #image_path = '/mnt/c/Unet/dataset5/track3/sport_normal/IMG/center_2020_09_12_16_51_19_594.jpg'
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate mask using SAM model
        sam_result = mask_generator.generate(image_rgb)
        segmentation = []


        for mask in sam_result:
           sorted(sam_result, key=lambda x: x['area'], reverse=True)

        #for mask in sam_result:
        #    segmentation.append(mask['segmentation'])
        for mask in sam_result:
            segmentation.append(mask['segmentation'])



        areas_as_strings = [str(mask['area']) for mask in sam_result]
        print(', '.join(areas_as_strings))
        

        masks = [
            mask['segmentation']
            for mask
            in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        ]
        '''
        masks = [
            mask['segmentation']
            for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)
            if mask['area'] == 18446
        ]

        
        sv.plot_images_grid(
            images=masks,
            grid_size=(100, int(len(masks) / 8)),
            size=(32, 32)
        )

        
        for img in segmentation:
            plt.imshow(img, cmap='gray')  # 'gray' colormap for black and white images
            plt.axis('off')  # Turn off axis
            plt.show()
        '''





        result = segmentation[0]

        mask_image = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] == 0:  # Schwarz
                    mask_image[i, j] = [0, 255, 0]  # Grün mask is in bgr
                else:  # Weiß
                    mask_image[i, j] = [255, 0, 0]  # Rot
        '''
        plt.imshow(mask_image)
        plt.show()
        '''


        annotated_image = cv2.addWeighted(image_bgr, 0.7, mask_image, 0.3, 0)

        '''
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Deaktiviere die Achsen
        plt.show()
        '''

        '''
        
        for row in result:
            for pixel in row:
                print(pixel, end=" ")
            print()
        '''


        sv.plot_images_grid(
            images = [result, image_bgr, annotated_image],
            grid_size=(1,3)

        )

        # Save annotated image with the same name in the saving path
        saving_filename = os.path.splitext(filename)[0] + "_mask.png"  # Change extension as needed
        saving_path = os.path.join(SAVING_PATH, saving_filename)
        #mask_image = (mask_image * 255).astype(np.uint8)


        cv2.imwrite(saving_path, mask_image)

        '''
        exit()
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Annotate image with masks
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        #show result for each img
        sv.plot_images_grid(
            images=[image_bgr, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )

        # Save annotated image with the same name in the saving path
        saving_filename = os.path.splitext(filename)[0] + "_mask.png"  # Change extension as needed
        saving_path = os.path.join(SAVING_PATH, saving_filename)
        mask_image = (annotated_image * 255).astype(np.uint8)
        cv2.imwrite(saving_path, mask_image)
        '''


''''

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


sam_result = mask_generator.generate(image_rgb)
print("finish generating")

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

print(annotated_image.shape)
print(detections.shape)

mask_image = (annotated_image * 255).astype(np.uint8)  # Convert to uint8 format
cv2.imwrite(SAVING_PATH, mask_image)



'''

