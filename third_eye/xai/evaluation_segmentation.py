from models.model2 import U_Net
import torch
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def preprocess(img):
    '''
    This function turning the input image into Unet model acceptable form
    '''
    # bei bedarf diese Zeile auskommentieren
    #img = img[:, :, :3]
    image = TF.to_tensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    return image


def result(predicted_rgb, seg):
    '''
    Calculate the percentage of correct predicted pixel
    '''
    all_pixel = predicted_rgb.shape[0] * predicted_rgb.shape[1]
    correct_prediction = 0
    for y in range(predicted_rgb.shape[0]):
        for x in range(predicted_rgb.shape[1]):
            if np.all(predicted_rgb[y, x] == (seg[y, x] * 255).astype(int)):
                correct_prediction += 1
    return correct_prediction / all_pixel


def evaluation(model, image_folder_path):
    files = os.listdir(image_folder_path)
    img_files = [f for f in files if f.startswith('2019')]

    for img_file in img_files:
        img_path = os.path.join(image_folder_path, img_file)

        img = mpimg.imread(img_path)

        x = preprocess(img)
        with torch.no_grad():
            prediction = model(x)


        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

        '''
        # change the color as the same as the mask img
        for y in range(seg.shape[0]):
            for x in range(seg.shape[1]):
                if np.all(predicted_rgb[y, x] == [1, 0, 255]):
                    predicted_rgb[y, x] = [0, 0, 255]
                if np.all(predicted_rgb[y, x] == [1, 255, 0]):
                    predicted_rgb[y, x] = [0, 255, 0]

        print(result(predicted_rgb, seg))
        '''

        a = 0
        if a % 2 == 0:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img)
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
    model.to(device)
    checkpoint_path = '/mnt/c/Unet/SegmentationModel2.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
#"C:\Unet\benchmark-ASE2022\icse20\DAVE2-Track1-DayNight\IMG"
    path = '/mnt/c/Unet/benchmark-ASE2022/icse20/DAVE2-Track1-DayNight/IMG/'

    evaluation(model, path)
