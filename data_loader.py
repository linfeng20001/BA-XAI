import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple
from tqdm import tqdm
import cv2 as cv2

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', augment=False):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'images', split)
        self.targets_dir = os.path.join(self.root, 'labels', split)
        self.split = split
        self.augment = augment
        self.images = []
        self.targets = []
        self.mapping = {
            0: 0,  # unlabeled
            0: 0,  # ego vehicle
            0: 0,  # rect border
            0: 0,  # out of roi
            0: 0,  # static
            77: 0,  # dynamic
            33: 0,  # ground
            90: 2,  # road
            120: 4,  # sidewalk
            193: 0,  # parking
            173: 0,  # rail track
            70: 3,  # building
            108: 9,  # wall
            164: 0,  # fence
            171: 0,  # guard rail
            115: 0,  # bridge
            126: 0,  # tunnel
            153: 6,  # pole
            153: 0,  # polegroup
            178: 0,  # traffic light
            195: 7,  # traffic sign
            119: 5,  # vegetation
            210: 11,  # terrain
            118: 10,  # sky
            84: 8,  # person
            76: 0,  # rider
            16: 1,  # car
            8: 0,  # truck
            47: 0,  # bus
            10: 0,  # caravan
            13: 0,  # trailer
            58: 0,  # train
            26: 0,  # motorcycle
            46: 0  # bicycle
        }
        self.mappingrgb = {
            0: (0, 0, 0),  # unlabeled
            0: (0, 0, 0),  # ego vehicle
            0: (0, 0, 0),  # rect border
            0: (0, 0, 0),  # out of roi
            0: (0, 0, 0),  # static
            77: (0, 0, 0),  # dynamic
            33: (0, 0, 0),  # ground
            90: (255, 0, 0),  # road
            120: (255, 0, 255),  # sidewalk
            193: (0, 0, 0),  # parking
            173: (0, 0, 0),  # rail track
            70: (0, 255, 0),  # building
            108: (153, 255, 0),  # wall
            164: (0, 0, 0),  # fence
            171: (0, 0, 0),  # guard rail
            115: (0, 0, 0),  # bridge
            126: (0, 0, 0),  # tunnel
            153: (255, 0, 153),  # pole
            153: (0, 0, 0),  # polegroup
            178: (0, 0, 0),  # traffic light
            195: (153, 0, 255),  # traffic sign
            119: (0, 255, 255),  # vegetation
            210: (0, 153, 153),  # terrain
            118: (255, 153, 0),  # sky
            84: (0, 153, 255),  # person
            76: (0, 0, 0),  # rider
            16: (0, 0, 255),  # car
            8: (0, 0, 0),  # truck
            47: (0, 0, 0),  # bus
            10: (0, 0, 0),  # caravan
            13: (0, 0, 0),  # trailer
            58: (0, 0, 0),  # train
            26: (0, 0, 0),  # motorcycle
            46: (0, 0, 0)  # bicycle
        }
        self.num_classes = 12
        # =============================================
        # Read in the paths to all images
        # =============================================
        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_color.png')
            self.targets.append(os.path.join(self.targets_dir, target_name))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Augment: {}\n'.format(self.augment)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):
        # Load the RGB image and target
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('L')

        # Apply random transformations if augmenting,
        # otherwise, just resize the image to the correct size
        if self.split == 'train':
            if self.augment:
                # Resize
                image = TF.resize(image, size=(256 + 20, 512 + 20), interpolation=Image.BILINEAR)
                target = TF.resize(target, size=(256 + 20, 512 + 20), interpolation=Image.NEAREST)
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 512))
                image = TF.crop(image, i, j, h, w)
                target = TF.crop(target, i, j, h, w)
                # Random horizontal flipping
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    target = TF.hflip(target)
                # Random vertical flipping (if needed)
            else:
                # Resize
                image = TF.resize(image, size=(256, 512), interpolation=Image.BILINEAR)
                target = TF.resize(target, size=(256, 512), interpolation=Image.NEAREST)
        if self.split == 'val':
            # Also resize for the validation/testing set
            image = TF.resize(image, size=(256, 512), interpolation=Image.BILINEAR)
            target = TF.resize(target, size=(256, 512), interpolation=Image.NEAREST)

        # Convert the image and target into PyTorch tensors
        image = TF.to_tensor(image)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))

        # Now normalize
        image = self.normalize(image)

        # Convert the target into the desired format
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = self.mask_to_rgb(target)
        targetrgb = targetrgb.long()

        # Finally, return the image pair
        return image, targetmask, targetrgb


if __name__ == '__main__':
    mask_image = cv2.imread(
        r'C:\Users\Linfe\OneDrive\Desktop\Seg\Dataset\labels\train\hamburg_000000_007737_gtFine_color.png',
        cv2.IMREAD_GRAYSCALE)


    # Zeige das Bild als 2D-Array an
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_image, cmap='gray')
    plt.colorbar()
    plt.show()
    print(mask_image)
    unique_values, value_counts = np.unique(mask_image, return_counts=True)

    # Gib die eindeutigen Graustufenwerte und ihre Häufigkeiten aus
    for value, count in zip(unique_values, value_counts):
        print(f"Graustufenwert: {value}, Anzahl: {count}")



##################################
    from PIL import Image

    img = Image.open(r'C:\Users\Linfe\OneDrive\Desktop\Seg\Dataset\labels\train\hamburg_000000_007737_gtFine_color.png').convert('L')  # convert image to 8-bit grayscale
    img.resize(size=(4, 4))
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].

    data_np = np.array(data, dtype=np.uint8)
'''
    # For example:
    for row in data:
        print(' '.join('{:3}'.format(value) for value in row))


    # Here's another more compact representation.
    chars = '@%#*+=-:. '  # Change as desired.
    scale = (len(chars) - 1) / 255.
    print()
    for row in data:
        print(' '.join(chars[int(value * scale)] for value in row))
'''


