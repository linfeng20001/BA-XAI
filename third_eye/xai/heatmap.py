import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.python.keras.models
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm

import torch
from models.model2 import U_Net
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

import ThirdEye.ase22.utils as utils
from ThirdEye.ase22.utils import *


#[  1   0 255]street [  1 255   0]background
def WeightedCombination(saliency_map, predicted_rgb, avg):
    for y in range(saliency_map.shape[1]):
        for x in range(saliency_map.shape[2]):
            if np.all(predicted_rgb[y, x] == [1, 255, 0]):
                saliency_map[0, y, x] = saliency_map[0, y, x] + avg
    average = np.average(saliency_map)
    return saliency_map, average
#only street attention are relevant
def MultiplicativeCombination(saliency_map, predicted_rgb):
    for y in range(saliency_map.shape[1]):
        for x in range(saliency_map.shape[2]):
            if np.all(predicted_rgb[y, x] == [1, 255, 0]):
                saliency_map[0, y, x] = 0
    average = np.average(saliency_map)
    return saliency_map, average



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
#heatmap and all the img will be saved in dataset5simulations and the csv is in asp benchmark
#all the heatmap score are in asp benchmark
def score_when_decrease(output):
    return -1.0 * output[:, 0]


def compute_heatmap(cfg, simulation_name, attention_type="SmoothGrad"):
    #prepare segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net(3, 2)
    model.to(device)

    checkpoint_path = '/mnt/c/Unet/SegmentationModel.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """
    Given a simulation by Udacity, the script reads the corresponding image paths from the csv and creates a heatmap for
    each driving image. The heatmap is created with the SmoothGrad algorithm available from tf-keras-vis
    (https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad). The scripts generates a separate
    IMG/ folder and csv file.
    """

    print("Computing attention heatmaps for simulation %s using %s" % (simulation_name, attention_type))

    # load the image file paths from csv


    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')

    path = "/mnt/c/Unet/track1/reverse/driving_log.csv"
    data_df = pd.read_csv(path)
    data = data_df["center"]
    
    '''
    img_path = '/mnt/c/Unet/dataset5/track1/normal/IMG'


    img_files = os.listdir(img_path)

    center_files = [file for file in img_files if "center" in file]
    data_df = pd.DataFrame({"center": center_files})
    #data_df = pd.read_csv(path)
    data = data_df["center"]
    '''


    print("read %d images from file" % len(data))
    #########################################################

    # load self-driving car model 
    self_driving_car_model = tensorflow.keras.models.load_model(
        Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    #SDC_PATH = '/mnt/c/Unet/udacity-dave2.h5'
    #self_driving_car_model = tensorflow.keras.models.load_model(SDC_PATH)

    #########################################################
    # load attention model
    saliency = None
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    elif attention_type == "GradCam++":
        saliency = GradcamPlusPlus(self_driving_car_model, model_modifier=None)

    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    total_time = 0
    prev_hm = gradient = np.zeros((80, 160))


    # create directory for the heatmaps
    '''
    path_save_heatmaps = os.path.join(cfg.TESTING_DATA_DIR,
                                      simulation_name,
                                      "heatmaps-" + attention_type.lower(),
                                      "IMG")
    
    if os.path.exists(path_save_heatmaps):
        print("Deleting folder at {}".format(path_save_heatmaps))
        shutil.rmtree(path_save_heatmaps)
    print("Creating image folder at {}".format(path_save_heatmaps))
    os.makedirs(path_save_heatmaps)
    '''
    path_save_heatmaps = '/mnt/c/Unet/dataset5simulations/heatmap'
    for idx, img in enumerate(tqdm(data)):
        #img = "/mnt/c/Unet/dataset5" + img

        # convert Windows path, if needed
        if "\\\\" in img:
            img = img.replace("\\\\", "/")
        elif "\\" in img:
            img = img.replace("\\", "/")

        # load image        x is for heatmap and y for segmentation
        x = y = image = mpimg.imread(img)

        # preprocess image
        ###################             why into 80 160 even smaller?
        x = utils.resize(x).astype('float32')

        #x = x.astype('float32')
        y = TF.to_tensor(y)

        #compute segmentation image

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        y = normalize(y)
        y = y.to(device)
        y = y.unsqueeze(0)

        #'''
        with torch.no_grad():
            prediction = model(y)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()
        #'''

        # compute heatmap image
        saliency_map = None
        if attention_type == "SmoothGrad":
            saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)



        # compute average of the heatmap
        #average = np.average(saliency_map)
        saliency_map, average = MultiplicativeCombination(saliency_map,predicted_rgb)

        # compute gradient of the heatmap
        if idx == 0:
            gradient = 0
        else:
            gradient = abs(prev_hm - saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map




        #saliency_map = np.stack((saliency_map,) * 3, axis=-1)#it turns into rgb but in graysacale for later to annotate with segmask

        '''
        saliency_map = np.squeeze(saliency_map)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

        
        # Display the first image on the first subplot
        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        # Display the second image (saliency_map) on the second subplot
        axes[1].imshow(saliency_map)
        axes[1].set_title('Saliency Map')

        axes[2].imshow(predicted_rgb)
        axes[2].set_title('SegMask')

        plt.show()
        
        exit()
        
'''
        # store the heatmaps
        file_name = img.split('/')[-1]
        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(path_save_heatmaps, file_name)
        #mpimg.imsave(path_name, np.squeeze(saliency_map))


        list_of_image_paths.append(path_name)

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)



    # save scores as numpy arrays
    file_name = "htm-" + attention_type.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg')

    np.save(path_name, avg_heatmaps)


    # plot scores as histograms
    plt.hist(avg_heatmaps)
    plt.title("average attention heatmaps")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-avg.png')
    plt.savefig(path_name)
    plt.show()

    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg-grad')
    np.save(path_name, avg_gradient_heatmaps)

    plt.clf()
    plt.hist(avg_gradient_heatmaps)
    plt.title("average gradient attention heatmaps")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-avg-grad.png')
    plt.savefig(path_name)
    plt.show()
    #TESTING_DATA_DIR = "/mnt/c/Unet/dataset5/track1/"
    # save as csv
    df = pd.DataFrame(list_of_image_paths, columns=['center'])
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    # data = data_df[["frameId", "time", "crashed"]]
    data = data_df[["frameId", "crashed"]]

    # copy frame id, simulation time and crashed information from simulation's csv
    df['frameId'] = data['frameId'].copy()
    # df['time'] = data['time'].copy()
    df['crashed'] = data['crashed'].copy()

    # save it as a separate csv
    df.to_csv(os.path.join(cfg.TESTING_DATA_DIR,
                           simulation_name,
                           "heatmaps-" + attention_type.lower(),
                           'driving_log.csv'), index=False)




if __name__ == '__main__':
    #ADS_MODEL_PATH = '/mnt/c/Unet/udacity-dave2.h5'
    cfg = Config()
    cfg.from_pyfile(filename="/mnt/c/Unet/ThirdEye/ase22/config_my.py")

    #simulation_name = "gauss-journal-track1-nominal"
    #simulation_name = 'mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1'
    simulation_name = 'reverse'
    '''
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    data = data_df["center"]
    for idx, img in enumerate(tqdm(data)):
        img = "/mnt/c/Unet/dataset5" + img
        x = mpimg.imread(img)
        plt.imshow(x) 

    '''


    compute_heatmap(cfg, simulation_name)

