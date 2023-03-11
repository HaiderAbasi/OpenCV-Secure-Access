import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from PIL import Image

import base64
import requests

from detection import Detector

# --------------------------------------------------
# configurations of dependencies

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])


if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image
    
detector = Detector()
    
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

# Face Recognition Pipeline: Step 1 & Step 2 (Detection & Alignment)
def extract_faces(
    img,
    detector_backend="opencv",
    grayscale=False,
    enforce_detection=True,
    align=True,
):
    target_size = (112, 112)

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == "skip":
        face_objs = [(img, img_region, 0)]
    else:
        face_objs = detector.detect_face(img, align)


    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        cv2.imshow("Face not detected",cv2.resize(img,(200,200)))
        cv2.waitKey(1)
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False.")

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:

            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            img_pixels = image.img_to_array(current_img)  # what this line doing? must?
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection == True:
        raise ValueError(
            f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
        )

    return extracted_faces

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
