# Noise Package


from skimage import img_as_float
import cv2 as cv
import numpy as np


class Noise:
    img = None
    float_flag = True

    def __init__(self, img):
        self.img = cv.imread(img)

    def gaussian_noise(self, mean=0, var=0.01):
        image = img_as_float(self.img)
        noise = np.random.normal(mean, var ** 0.5,
                                 image.shape)
        out = image + noise
        return out

    def salt_and_pepper(self, amount=0.05, salt_vs_pepper=0.25):
        image = img_as_float(self.img)

        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        out = image.copy()
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[amount, 1 - amount])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[salt_vs_pepper, 1 - salt_vs_pepper])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = low_clip
        return out

    def salt(self):
        return self.salt_and_pepper(salt_vs_pepper=1.)

    def pepper(self):
        return self.salt_and_pepper(salt_vs_pepper=0.)

    def poisson(self):

        image = img_as_float(self.img)
        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))

        old_max = None
        if low_clip == -1.:
            old_max = image.max()
            image = (image + 1.) / (old_max + 1.)

        out = np.random.poisson(image * vals) / float(vals)
        if low_clip == -1.:
            out = out * (old_max + 1.) - 1.
        return out

    def mean_noise(self, mean=0, var=0.01):
        image = img_as_float(self.img)
        noise = np.random.normal(mean, var ** 0.5,
                                 image.shape)
        out = image + image * noise
        return out

    def gaussian_and_salt_and_pepper(self, mean=0, var=0.01, amount=0.02, salt_vs_pepper=0.1):
        another = self.img.copy()
        out = self.gaussian_noise(mean=mean, var=var)
        self.img = out
        out = self.salt_and_pepper(amount=amount, salt_vs_pepper=salt_vs_pepper)
        self.img = another
        return out


