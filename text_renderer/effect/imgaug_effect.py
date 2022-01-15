from typing import Tuple

from PIL import Image
from imgaug.augmenters import Augmenter
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import numpy as np

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class ImgAugEffect(Effect):
    """
    Apply imgaug(https://github.com/aleju/imgaug) Augmenter on image.
    """

    def __init__(self, p=1.0, aug: Augmenter = None):
        super().__init__(p)
        self.aug = aug

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        if self.aug is None:
            return img, text_bbox

        word_img = np.array(img)
        # TODO: test self.aug.augment_bounding_boxes()

        auged=None
        try:
            auged = self.aug.augment_image(word_img)
        except Exception as e:
            # Resize to 32:
            if('at least 32 pixels' in str(e)):
                expand_ratio = 32 / word_img.shape[0]
                new_w = int(word_img.shape[1] * expand_ratio)
                new_h = 32
                word_img = np.array(img.resize((new_w, new_h), Image.ANTIALIAS))
                # auged = self.aug.augment_image(word_img)

            # if(('Expected input image to have shape (height, width) or (height, width, 1) or (height, width, 3)') in str(e)):
            try:
                img_rgb = word_img[:,:,0:3]
                img_alpha = word_img[:,:,3]
                auged = self.aug.augment_image(img_rgb)
                auged_rgba = np.dstack((auged,img_alpha))
                auged = auged_rgba
            except:
                auged = word_img



        return Image.fromarray(auged), text_bbox


        word_img = np.array(img)
        # TODO: test self.aug.augment_bounding_boxes()
        return Image.fromarray(self.aug.augment_image(word_img)), text_bbox




class Emboss(ImgAugEffect):
    def __init__(self, p=1.0, alpha=(0, 9, 1.0), strength=(1.5, 1.6)):
        """

        Parameters
        ----------
        p:
        alpha:
            see imgaug `doc`_
        strength:
            see imgaug `doc`_


        .. _doc: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html#imgaug.augmenters.convolutional.Emboss
        """
        super().__init__(p, iaa.Emboss(alpha=alpha, strength=strength))


class MotionBlur(ImgAugEffect):
    def __init__(self, p=1.0, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)):
        """

        Parameters
        ----------
        p
        k: 
            Kernel size to use.
        angle
            Angle of the motion blur in degrees (clockwise, relative to top center direction).
        direction
            Forward/backward direction of the motion blur. Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle). 
            Higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.


        .. _doc: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html#imgaug.augmenters.blur.MotionBlur
        """

        super().__init__(p, iaa.MotionBlur(k, angle, direction))





class SaltAndPepper(ImgAugEffect):
    def __init__(self, p=1.0, noise=0.1):
        super().__init__(p, iaa.SaltAndPepper(noise))


class CoarseDropout(ImgAugEffect):
#https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#coarsedropout
    def __init__(self, p=1.0,noise=0.02, size_percent=0.5):
        super().__init__(p, iaa.CoarseDropout(noise,size_percent=size_percent))

class Snow(ImgAugEffect):
# https://imgaug.readthedocs.io/en/latest/source/overview/imgcorruptlike.html#snow
    def __init__(self, p=1.0,level=1):
        super().__init__(p, iaa.imgcorruptlike.Snow(severity=level))

class JpegCompression(ImgAugEffect):
# https://imgaug.readthedocs.io/en/latest/source/overview/imgcorruptlike.html#snow
    def __init__(self, p=1.0,level=1):
        super().__init__(p, iaa.imgcorruptlike.JpegCompression(severity=level))


class SnowFlakes(ImgAugEffect):
# https://imgaug.readthedocs.io/en/latest/source/overview/weather.html#snowflakes
    def __init__(self, p=1.0,flake_size=(0.1, 0.4), speed=(0.01, 0.05)):
        super().__init__(p, iaa.Snowflakes(flake_size=flake_size, speed=speed))





