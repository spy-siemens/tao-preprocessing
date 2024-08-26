import tensorflow as tf
# def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=(224, 224),
#                       interpolation='bilinear:random', keep_aspect_ratio=False):
#     """Wraps keras_preprocessing.image.utils.load_img() and adds cropping.
#
#     Cropping method enumarated in interpolation
#     # Arguments
#         path: Path to image file.
#         color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
#             The desired image format.
#         target_size: Either `None` (default to original size)
#             or tuple of ints `(img_height, img_width)`.
#         interpolation: Interpolation and crop methods used to resample and crop the image
#             if the target size is different from that of the loaded image.
#             Methods are delimited by ":" where first part is interpolation and second is crop
#             e.g. "lanczos:random".
#             Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
#             "box", "hamming" By default, "nearest" is used.
#             Supported crop methods are "none", "center", "random".
#     # Returns
#         A PIL Image instance.
#     # Raises
#         ImportError: if PIL is not available.
#         ValueError: if interpolation method is not supported.
#     """
#     # Decode interpolation string. Allowed Crop methods: none, center, random
#     interpolation, crop = interpolation.split(":") \
#         if ":" in interpolation else (interpolation, "none")
#
#     if crop == "none":
#         print("crop none", interpolation)
#         return load_img(
#             path,
#             color_mode=color_mode,
#             target_size=target_size,
#             interpolation=interpolation)
import random
from config import _PIL_INTERPOLATION_METHODS, COLOR_AUGMENTATION
from augmentation import color_augmentation

def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest', keep_aspect_ratio=False):
    """Wraps keras_preprocessing.image.utils.load_img() and adds cropping.

    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") \
        if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return tf.keras.preprocessing.image.load_img(
            path,
            grayscale=grayscale,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)

    # Load original size image using Keras
    img = tf.keras.preprocessing.image.load_img(
        path,
        grayscale=grayscale,
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
                return img
            raise ValueError("Crop mode not supported.")

    return img
