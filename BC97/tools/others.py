import os
import cv2

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def OU(x, mu, sigma, theta):
    """
    x, mu, sigma, theta
    """
    return theta * (mu - x) + sigma * np.random.randn(1)


def render_image(name, img):
    """using cv2
    
    Arguments:
        name {str}
        img {np.array}
    """
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(1)



import csv

def create_csv(path, csv_head):

    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

def write_csv(path, data_row):

    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)



ia.seed(1)

class ImgAug:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SomeOf((1, 3),
            [
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Dropout((0.0, 0.01), per_channel=0.2)
            ])
        ], random_order=True) # apply augmenters in random order

    def process(self, imgs):
        return self.seq(imgs)