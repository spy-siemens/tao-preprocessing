import io, pathlib, random
from PIL import Image as pil_image
import numpy as np
import cv2

pil_image_resampling = pil_image.Resampling
_PIL_INTERPOLATION_METHODS = {
    "nearest": pil_image_resampling.NEAREST,
    "bilinear": pil_image_resampling.BILINEAR,
    "bicubic": pil_image_resampling.BICUBIC,
    "hamming": pil_image_resampling.HAMMING,
    "box": pil_image_resampling.BOX,
    "lanczos": pil_image_resampling.LANCZOS,
}

CROP_PADDING = 32
COLOR_AUGMENTATION = False


def load_and_crop_img(path, color_mode='rgb', target_size=(224, 224), interpolation='bilinear'):
    # TODO: investigate which crop was used, assume to be none
    interpolation, crop = interpolation, "none"

    if crop == "none":
        return load_img(path, color_mode=color_mode, target_size=target_size, interpolation=interpolation)


def _preprocess_numpy_input(x, data_format="channels_last", mode="caffe", color_mode="rgb", img_mean=None, img_depth=8,
                            **kwargs):
    if color_mode == "rgb":
        assert img_depth == 8, (
            f"RGB images only support 8-bit depth, got {img_depth}, "
            "please check `model.input_image_depth` in spec file"
        )
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]

    for idx in range(len(mean)):
        x[..., idx] -= mean[idx]
    return x


def load_img(path, color_mode="rgb", target_size=None, interpolation="nearest", keep_aspect_ratio=False):
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. The use of `load_img` requires PIL."
        )
    if isinstance(path, str):
        if isinstance(path, io.BytesIO):
            img = pil_image.open(path)
        elif isinstance(path, (pathlib.Path, bytes, str)):
            if isinstance(path, pathlib.Path):
                path = str(path.resolve())
            with open(path, "rb") as f:
                img = pil_image.open(io.BytesIO(f.read()))
        else:
            raise TypeError(
                f"path should be path-like or io.BytesIO, not {type(path)}"
            )
    else:
        get_image_from_vision_payload(path)

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys()),
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img


def load_and_crop_img_v2(path, color_mode='rgb', target_size=(224, 224), interpolation='bilinear:center'):
    interpolation, crop = interpolation.split(":") \
        if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return load_img(path,
                        color_mode=color_mode,
                        target_size=target_size,
                        interpolation=interpolation)

    # Load original size image using Keras
    img = load_img(path,
                   color_mode=color_mode,
                   target_size=None,
                   interpolation=interpolation)

    # Crop fraction of total image
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError(f'Invalid crop method {crop} specified.')

            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '  # noqa pylint: disable=C0209
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(
                            _PIL_INTERPOLATION_METHODS.keys())))

            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            if crop == 'random':
                # Resize keeping aspect ratio
                # result should be no smaller than the targer size, include crop fraction overhead
                crop_fraction = random.uniform(0.45, 1.0)
                target_size_before_crop = (
                    target_width / crop_fraction,
                    target_height / crop_fraction
                )
                ratio = max(
                    target_size_before_crop[0] / width,
                    target_size_before_crop[1] / height
                )
                target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
                img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            if crop == 'center':
                # Resize keeping aspect ratio
                # result should be no smaller than the larger size, include crop fraction overhead
                target_size_before_crop = (
                    target_width + CROP_PADDING,
                    target_height + CROP_PADDING
                )
                ratio = max(
                    target_size_before_crop[0] / width,
                    target_size_before_crop[1] / height
                )
                target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
                img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width / 2)) - int(round(target_width / 2))
                top_corner = int(round(height / 2)) - int(round(target_height / 2))
                return img.crop(
                    (left_corner,
                     top_corner,
                     left_corner + target_width,
                     top_corner + target_height))
            if crop == "random":
                # random crop
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                img = img.crop(
                    (left_shift,
                     down_shift,
                     target_width + left_shift,
                     target_height + down_shift))
                # color augmentation
                if COLOR_AUGMENTATION and img.mode == "RGB":
                    return color_augmentation(img)
                    pass
                return img
            raise ValueError("Crop mode not supported.")

    return img


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
    return pil_image.fromarray(x_img.astype(np.uint8), "RGB")


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


def random_saturation(img, max_shift):
    """random saturation data augmentation."""
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    shift = randu(-max_shift, max_shift)
    # saturation should always be within [0,1.0]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + shift, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_contrast(img, center, max_contrast_scale):
    """random contrast data augmentation."""
    new_img = (img - center) * (1.0 + randu(-max_contrast_scale, max_contrast_scale)) + center
    new_img = np.clip(new_img, 0., 1.)
    return new_img


def random_shift(x_img, shift_stddev):
    """random shift data augmentation."""
    shift = np.random.randn() * shift_stddev
    new_img = np.clip(x_img + shift, 0.0, 1.0)

    return new_img


def randu(low, high):
    """standard uniform distribution."""
    return np.random.random() * (high - low) + low


def get_image_from_vision_payload(data: dict):
    image = data["detail"][0]
    width = image["width"]
    height = image["height"]
    image_bytes = image["image"]
    pil_image_read = pil_image.frombytes("RGB", (width, height), image_bytes)
    return pil_image_read


def generate_input(img_path: str):
    cropped_image = load_and_crop_img_v2(img_path)
    np_cropped_image = np.array(cropped_image).astype(np.float32)
    infer_input = _preprocess_numpy_input(np_cropped_image)
    infer_input = infer_input.transpose(2, 0, 1)
    infer_input = infer_input[None, ...]
    return infer_input
