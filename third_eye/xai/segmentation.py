import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.python.keras.models
from matplotlib import image as mpimg
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
import re
import torch

from models.model2 import U_Net
import torchvision.transforms.functional as TF
from torchvision import transforms
import ThirdEye.ase22.utils as utils
import cv2


def preprocessForSegmentation(img):
    '''
    This function turning the input image into Unet model acceptable form
    '''
    # bei bedarf diese Zeile auskommentieren
    # img = img[:, :, :3]
    image = TF.to_tensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)
    image = image.to('cpu')
    image = image.unsqueeze(0)
    return image


def MultiplicativeCombination(saliency_map, predicted_rgb):
    global_attention = road_attention = road_pixel = 0
    for x in range(saliency_map.shape[0]):
        for y in range(saliency_map.shape[1]):
            global_attention += saliency_map[x, y]
            if np.all(predicted_rgb[x, y] == [0, 255, 0]):
                saliency_map[x, y] = 0

    for x in range(saliency_map.shape[0]):
        for y in range(saliency_map.shape[1]):
            if (saliency_map[x, y] != 0):
                road_pixel += 1
                road_attention += saliency_map[x, y]
    road_attention_average = road_attention / road_pixel
    road_attention_percentage = road_attention / global_attention * 100
    return saliency_map, road_attention_average, road_attention_percentage


mapping = {
    149: 0,
    29: 1
}
mappingrgb = {
    29: (0, 0, 255),
    149: (0, 255, 0)
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


def score_when_decrease(output):
    return -1.0 * output[:, 0]


def comute_segmentation(cfg, simulation_name):
    #pre seg model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net(3, 2)
    model.to(device)

    checkpoint_path = '/mnt/c/Unet/SegmentationModel4.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    #load img

    path = os.path.join(cfg.TESTING_DATA_DIR,  # 'C:/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022/mutants'
                        simulation_name,  #
                        'driving_log.csv')

    data_df = pd.read_csv(path)
    # data = data_df["center"]
    data = data_df["center"].apply(
        lambda x: "/mnt/c/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022" + x.replace("simulations", ""))

    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    total_time = 0
    prev_hm = gradient = np.zeros((80, 160))

    for idx, img in enumerate(tqdm(data)):
        if "\\\\" in img:
            img = img.replace("\\\\", "/")
        elif "\\" in img:
            img = img.replace("\\", "/")

        # prepare heatmap
        file_name = img.split('/')[-1]
        heatmap_path = cfg.TESTING_DATA_DIR + '/' + simulation_name + '/heatmaps-smoothgrad/IMG/htm-smoothgrad-' + file_name

        image = mpimg.imread(img)

        y = preprocessForSegmentation(image)

        # prediction is in tensor
        with torch.no_grad():
            prediction = model(y)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

        heatmap_image = mpimg.imread(heatmap_path)
        #preprocess heatmap again into attention score
        #heatmap_image =


if __name__ == '__main__':
    device = 'cpu'
    model = U_Net(3, 2)
    model.to(device)

    checkpoint_path = 'C:/Unet/temp/SegmentationModel_CrossEntropyLoss12.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    img = "C:/Unet/benchmark-ASE2022/gauss-journal-track1-nominal/IMG/2021_02_09_19_47_53_058.jpg"
    image = mpimg.imread(img)

    crop = True
    yuv = True
    size = False
    if crop:
        image = utils.crop(image)
    if size:
        image = utils.resize(image)
    else:
        image = cv2.resize(image, (320, 160), cv2.INTER_AREA)


    plt.imshow(image)
    plt.show()
    y = preprocessForSegmentation(image)

    # prediction is in tensor
    with torch.no_grad():
        prediction = model(y)

    predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
    maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
    predicted_rgb = class_to_rgb(maxindex).to('cpu')
    predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

    plt.imshow(predicted_rgb)
    plt.show()