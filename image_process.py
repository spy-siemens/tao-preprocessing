import io, pathlib
from PIL import Image as pil_image
import numpy as np


pil_image_resampling = pil_image.Resampling
_PIL_INTERPOLATION_METHODS = {
    "nearest": pil_image_resampling.NEAREST,
    "bilinear": pil_image_resampling.BILINEAR,
    "bicubic": pil_image_resampling.BICUBIC,
    "hamming": pil_image_resampling.HAMMING,
    "box": pil_image_resampling.BOX,
    "lanczos": pil_image_resampling.LANCZOS,
}


def load_and_crop_img(path, color_mode='rgb', target_size=(224, 224), interpolation='bilinear', keep_aspect_ratio=False):
    #TODO: investigate which crop was used, asume to be none
    interpolation, crop = interpolation, "none"

    if crop == "none":
        return load_img(path, color_mode=color_mode, target_size=target_size, interpolation=interpolation)


def _preprocess_numpy_input(x, data_format="channels_last", mode="caffe", color_mode="rgb", img_mean=None, img_depth=8, **kwargs):

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


def generate_input(img_path: str):
    cropped_image = load_and_crop_img(img_path)
    np_cropped_image = np.array(cropped_image).astype(np.float32)
    infer_input = _preprocess_numpy_input(np_cropped_image)
    infer_input = infer_input.transpose(2, 0, 1)
    infer_input = infer_input[None, ...]
    return infer_input
