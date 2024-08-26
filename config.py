from PIL import Image as pil_image
pil_image_resampling = pil_image.Resampling
_PIL_INTERPOLATION_METHODS = {
    "nearest": pil_image_resampling.NEAREST,
    "bilinear": pil_image_resampling.BILINEAR,
    "bicubic": pil_image_resampling.BICUBIC,
    "hamming": pil_image_resampling.HAMMING,
    "box": pil_image_resampling.BOX,
    "lanczos": pil_image_resampling.LANCZOS,
}

# Constants from the configuration
DATA_FORMAT = 'channels_first'
NUM_CLASSES = 5
IMAGE_MEAN = [103.939, 116.779, 123.68]
PREPROCESS_MODE = 'caffe'
INPUT_CHANNELS = 3
IMG_HEIGHT = 224
IMG_WIDTH = 224
TARGET_SIZE = [IMG_HEIGHT, IMG_WIDTH]
IMG_DEPTH = 8
INTERPOLATION = 'bilinear'
BATCH_SIZE = 64
RANDOM_SEED = 42

COLOR_AUGMENTATION = False
# Label mapping
LABEL_MAP = {0: "ET200AL", 1: "ET200ecoPN", 2: "ET200sp", 3: "S7_1200", 4: "S7_1500"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}