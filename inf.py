import numpy as np
import torch
from models.model2 import U_Net
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision import transforms

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_Net(3, 2)
model.to(device)


checkpoint_path = '/mnt/c/Unet/SegmentationModel.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
#'/mnt/c/Unet/benchmark-ASE2022/gauss-journal-track1-nominal/IMG/2021_02_09_19_47_53_058.jpg'
#"C:\Unet\benchmark-ASE2022\mutants\udacity_add_weights_regularisation_mutated0_MP_l1_3_1\IMG\2022_04_21_13_11_45_057.jpg"
image_path = '/mnt/c/Unet/benchmark-ASE2022/mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1/IMG/2022_04_21_13_11_45_057.jpg'
mask_path = '/mnt/c/Unet/new_dataset/labels/train/road_2024_03_01_13_48_03_003.png'
heatmap = '/mnt/c/Unet/dataset5simulations/heatmap/htm-smoothgrad-2021_02_09_19_47_57_249.jpg'



original_image = Image.open(image_path).convert('RGB')
target = Image.open(mask_path).convert('L')
heatmap_img = Image.open(heatmap)


#76 and 150


image = TF.to_tensor(original_image)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image = normalize(image)
image = image.to(device)
image = image.unsqueeze(0)

with torch.no_grad():
    prediction = model(image)


predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
predicted_rgb = class_to_rgb(maxindex).to('cpu')


predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

height, width = heatmap_img.size

'''
for y in range(height):
    for x in range(width):
        print(predicted_rgb[x,y])
        '''


plt.imshow(original_image)
plt.show()
plt.imshow(predicted_rgb)
plt.show()



#green is background and blue is road



'''
plt.imshow(original_image)
plt.show()

'''
