from numba import jit, njit
import numpy as np
import cv2
from PIL import Image


@njit
def randu(low, high):
    """standard uniform distribution."""
    return np.random.random() * (high - low) + low
@jit
def random_hue(img, max_delta=10.0):
    """Rotates the hue channel.

    Args:
        img: input image in float32
        max_delta: Max number of degrees to rotate the hue channel
    """
    # Rotates the hue channel by delta degrees
    delta = randu(-max_delta, max_delta)
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hchannel = hsv[:, :, 0]
    hchannel = delta + hchannel
    # hue should always be within [0,360]
    idx = np.where(hchannel > 360)
    hchannel[idx] = hchannel[idx] - 360
    idx = np.where(hchannel < 0)
    hchannel[idx] = hchannel[idx] + 360
    hsv[:, :, 0] = hchannel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@jit
def random_saturation(img, max_shift):
    """random saturation data augmentation."""
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    shift = randu(-max_shift, max_shift)
    # saturation should always be within [0,1.0]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + shift, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@jit
def random_contrast(img, center, max_contrast_scale):
    """random contrast data augmentation."""
    new_img = (img - center) * (1.0 + randu(-max_contrast_scale, max_contrast_scale)) + center
    new_img = np.clip(new_img, 0., 1.)
    return new_img


@jit
def random_shift(x_img, shift_stddev):
    """random shift data augmentation."""
    shift = np.random.randn() * shift_stddev
    new_img = np.clip(x_img + shift, 0.0, 1.0)

    return new_img


def color_augmentation(
    x_img,
    color_shift_stddev=0.0,
    hue_rotation_max=25.0,
    saturation_shift_max=0.2,
    contrast_center=0.5,
    contrast_scale_max=0.1
):
    """color augmentation for images."""
    # convert PIL Image to numpy array
    x_img = np.array(x_img, dtype=np.float32)
    # normalize the image to (0, 1)
    x_img /= 255.0
    x_img = random_shift(x_img, color_shift_stddev)
    x_img = random_hue(x_img, max_delta=hue_rotation_max)
    x_img = random_saturation(x_img, saturation_shift_max)
    x_img = random_contrast(
        x_img,
        contrast_center,
        contrast_scale_max
    )
    # convert back to PIL Image
    x_img *= 255.0
    return Image.fromarray(x_img.astype(np.uint8), "RGB")