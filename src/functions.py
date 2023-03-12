import os
import cv2
import numpy as np
import base64
import requests

from pathlib import Path
from PIL import Image

    
def loadBase64Img(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    exact_image = False
    base64_img = False
    url_img = False

    if type(img).__module__ == np.__name__:
        exact_image = True

    elif img.startswith("data:image/"):
        base64_img = True

    elif img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img is True:
        img = loadBase64Img(img)

    elif url_img is True:
        img = np.array(Image.open(requests.get(img, stream=True, timeout=60).raw).convert("RGB"))

    elif exact_image is not True:  # image path passed as input
        if os.path.isfile(img) is not True:
            raise ValueError(f"Confirm that {img} exists")

        img = cv2.imread(img)

    return img


# --------------------------------------------------

def normalize_input(img, normalization="base"):

    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if normalization == "raw":
        pass  # return just restored pixels

    elif normalization == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")

    return img

# --------------------------------------------------

def initialize_folder():
    home = get_arcface_home()

    if not os.path.exists(home + "/.arcface"):
        os.makedirs(home + "/.arcface")
        print("Directory ", home, "/.arcface created")

    if not os.path.exists(home + "/.arcface/weights"):
        os.makedirs(home + "/.arcface/weights")
        print("Directory ", home, "/.arcface/weights created")


def get_arcface_home():
    return str(os.getenv("ARCFACE_HOME", default=str(Path.home())))

# --------------------------------------------------
