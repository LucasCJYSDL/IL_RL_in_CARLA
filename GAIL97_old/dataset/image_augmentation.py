import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import time

ia.seed(1)

# images = np.array(
#     [ia.quokka(size=(88, 200)) for _ in range(128)],
#     dtype=np.uint8
# )

seq = iaa.Sequential([

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

# str = time.time()
# images_aug = seq(images=images)
# end = time.time()
#
# print("time: ", end-str)