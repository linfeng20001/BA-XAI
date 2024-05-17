from models.model2 import U_Net
import torch
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
from ThirdEye.ase22.utils import resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapping = {
    150: 0,
    76: 1
}
mappingrgb = {
    76: (1, 0, 255),
    150: (1, 255, 0)
}


def pixelwise_accuracy(img, seg):
    assert img.shape == seg.shape, "Images must have the same shape"

# Flatten images to 1D arrays
    img_flat = img.ravel()
    seg_flat = seg.ravel()

# Count pixels that match
    correct_pixels = np.sum(img_flat == seg_flat)

# Calculate accuracy
    accuracy = correct_pixels / len(img_flat)

    return accuracy

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


def preprocess(img):
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


def result(predicted_rgb, seg):
    '''
    Calculate the percentage of correct predicted pixel
    '''
    predicted_rgb = resize(predicted_rgb)
    seg = resize(seg)
    all_pixel = predicted_rgb.shape[0] * predicted_rgb.shape[1]
    correct_prediction = 0
    for y in range(predicted_rgb.shape[0]):
        for x in range(predicted_rgb.shape[1]):
            if np.array_equal(predicted_rgb[y, x], [1, 255, 0]) and np.array_equal(seg[y, x], [0, 1, 0]):
                correct_prediction += 1
            elif np.array_equal(predicted_rgb[y, x], [1, 0, 255]) and np.array_equal(seg[y, x], [0, 0, 1]):
                correct_prediction += 1
    return correct_prediction / all_pixel*100


def evaluation(model, image_folder_path):
    files = os.listdir(image_folder_path)
    img_files = [f for f in files if f.startswith('image')]

    for img_file in img_files:
        img_path = os.path.join(image_folder_path, img_file)

        img = mpimg.imread(img_path)

        segmentation_name = "segmentation" + img_file[5:-3] + "png"
        segmentation_path = os.path.join(image_folder_path, segmentation_name)
        seg = mpimg.imread(segmentation_path)


        x = preprocess(img)
        with torch.no_grad():
            prediction = model(x)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()
        '''
        for y in range(seg.shape[0]):
            for x in range(seg.shape[1]):
                if np.all(seg[y, x] == [0,1,0]):
                    seg[y, x] = [1,255,0]
                elif np.all(seg[y, x] == [0,0,1]):
                    seg[y, x] = [1, 0, 255]
        '''

        #print(pixelwise_accuracy(predicted_rgb,seg))

        print(result(predicted_rgb, seg))

        a = 0
        if a % 2 == 0:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(seg)
            axs[0].set_title('Image')
            axs[0].axis('off')

            axs[1].imshow(predicted_rgb)
            axs[1].set_title('prediction')
            axs[1].axis('off')
            plt.show()

        a += 1


if __name__ == '__main__':
    import os

    model = U_Net(3, 2)
    model.to('cpu')
    checkpoint_path = 'C:/Unet/SegmentationModel_CrossEntropyLoss2.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    # "C:\Unet\benchmark-ASE2022\icse20\DAVE2-Track1-DayNight\IMG"

    path = 'C:/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022/mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1/IMG/'
    files = os.listdir(path)
    for img in files:
        image = mpimg.imread(os.path.join(path, img))

        plt.imshow(image)
        plt.show()
        image = preprocess(image)
        with torch.no_grad():
            prediction = model(image)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()



        plt.imshow(predicted_rgb)
        plt.show()

    exit()

    path = 'C:/Unet/new_segmentation_dataset/test'

    evaluation(model, path)
