import torch
from models.model import Unet
from torchvision import transforms
from data_loader.data_loader import Dataset
from utils.visualize import Visualizer  # Ensure this is correctly imported
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import time


def measure_inference_speed(model, image_path, num_iterations=40):
    total_time = 0.0
    original_image = Image.open(image_path).convert('RGB')
    original_image = TF.resize(original_image, size=(256, 512), interpolation=Image.BILINEAR)
    image = TF.to_tensor(original_image)
    image = normalize(image)
    image = image.unsqueeze(0)

    for counter in range(num_iterations):
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            prediction = model(image)
        end_time = time.time()

        # Accumulate total inference time
        total_time += end_time - start_time
        print(counter)

    # Calculate average inference time
    average_time = total_time / num_iterations
    print("Average inference time per image: {:.4f} seconds".format(average_time))


mapping = {
    150: 0,
    76: 1
}
mappingrgb = {
    76: (1, 0, 255),
    150: (1, 255, 0)
}


def class_to_rgb(mask):
    '''
    This function maps the classification index ids into the rgb.
    For example after the argmax from the network, you want to find what class
    a given pixel belongs too. This does that but just changes the color
    so that we can compare it directly to the rgb groundtruth label.
    '''
    mask2class = dict((v, k) for k, v in mapping.items())
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mask2class:
        rgbimg[0][mask == k] = mappingrgb[mask2class[k]][0]
        rgbimg[1][mask == k] = mappingrgb[mask2class[k]][1]
        rgbimg[2][mask == k] = mappingrgb[mask2class[k]][2]
    return rgbimg


def mask_to_class(mask):
    '''
    Given the cityscapes dataset, this maps to a 0..classes numbers.
    This is because we are using a subset of all masks, so we have this "mapping" function.
    This mapping function is used to map all the standard ids into the smaller subset.
    '''
    maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mapping:
        maskimg[mask == k] = mapping[k]
    return maskimg


def mask_to_rgb(mask):
    '''
    Given the Cityscapes mask file, this converts the ids into rgb colors.
    This is needed as we are interested in a sub-set of labels, thus can't just use the
    standard color output provided by the dataset.
    '''
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mappingrgb:
        rgbimg[0][mask == k] = mappingrgb[k][0]
        rgbimg[1][mask == k] = mappingrgb[k][1]
        rgbimg[2][mask == k] = mappingrgb[k][2]
    return rgbimg


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(num_classes=12, input_channels=3, num_filters=32, Dropout=0.3, res_blocks_dec=True)
model.to(device)

# Load the trained weights into the model
checkpoint_path = '/mnt/c/Unet/unet.pkl'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
# "C:\Unet\new_dataset\images\train\center_2024_03_01_13_46_05_282.png"
# Path to the image you want to use for inference
image_path = '/mnt/c/Unet/new_dataset/images/train/center_2024_03_01_13_46_05_282.png'
mask_path = '/mnt/c/Unet/new_dataset/labels/train/road_2024_03_01_13_46_05_282.png'
original_image = Image.open(image_path).convert('RGB')
target = Image.open(mask_path).convert('L')

original_image = TF.resize(original_image, size=(256, 512), interpolation=Image.BILINEAR)
target = TF.resize(target, size=(256, 512), interpolation=Image.NEAREST)

image = TF.to_tensor(original_image)
target = torch.from_numpy(np.array(target, dtype=np.uint8))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image = normalize(image)

targetmask = mask_to_class(target)
targetmask = targetmask.long()
targetrgb = mask_to_rgb(target)
targetrgb = targetrgb.long()

image = image.to(device)
image = image.unsqueeze(0)

# Perform inference
with torch.no_grad():
    prediction = model(image)

predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to(device)
maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
predicted_rgb = class_to_rgb(maxindex).to(device)

targetrgb = targetrgb.float() / 255.0 if targetrgb.dtype == torch.uint8 else targetrgb.float()

# Instantiate the Visualizer
visualizer = Visualizer()
# Visualize the result using the Visualizer class
visualizer.show_images(original_image, targetrgb, predicted_rgb)

# time measure
measure_inference_speed(model, image_path)
