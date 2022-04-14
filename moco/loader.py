# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
#for more noise transformations
import numpy as np
from skimage.util import random_noise
from PIL import Image
#for JPEG compression
from io import BytesIO


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class ThreeCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, base_transform_mild):
        self.base_transform = base_transform
        self.base_transform_mild = base_transform_mild

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        r = self.base_transform_mild(x)
        return [q, k, r]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
########################################################
#JPEG compression artifacts
########################################################

class JPEGCompress(object):
    def __init__(self,  quality=10, **kwargs):
        self.quality = quality
        self.kwargs = kwargs
    def __call__(self, x):
        buffer = BytesIO()
        x.save(buffer, "JPEG", quality=self.quality)
        x2 = Image.open(buffer)
        return x2
########################################################
#more random_noise transformations from skimage
########################################################
class RandomNoiseGrayscale(object):
    """skimage random noise applied to PIL Images for grayscale images represented as full RGB Images"""

    def __init__(self,  mode='gaussian', seed=None, clip=True, **kwargs):
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def __call__(self, x):
        return Image.fromarray((255*random_noise(np.asarray(x)[:,:,0], mode=self.mode,seed=self.seed,clip=self.clip,**self.kwargs)).astype('uint8')).convert("RGB")

class RandomNoiseRGB(object):
    """skimage random noise applied to PIL Images"""

    def __init__(self,  mode='gaussian', seed=None, clip=True, **kwargs):
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def __call__(self, x):
        return Image.fromarray((255*random_noise(np.asarray(x), mode=self.mode,seed=self.seed,clip=self.clip,**self.kwargs)).astype('uint8'))
