import time
import distance as dst
import numpy as np
import cv2

import ArcFace

from functions import initialize_folder,load_image,normalize_input

from detection import Detector

# configurations of dependencies
import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image
    


class FaceRecognizer:
    
    def __init__(self):
        # Intiialzing home folder for keeping arcface model and weights
        initialize_folder()
        
        # !) Creating Detector object for face detecion and alignment
        self.detector = Detector()

        # 2) Loading Arcface model for face representation
        self.model_obj = ArcFace.loadModel()
    
    # Face Recognition Pipeline: Step 1 & Step 2 (Detection & Alignment)
    def extract_faces(self, img, detector_backend="opencv",
                      grayscale=False, enforce_detection=True, align=True):
        
        target_size = (112, 112)

        # this is going to store a list of img itself (numpy), it region and confidence
        extracted_faces = []

        # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
        img = load_image(img)
        img_region = [0, 0, img.shape[1], img.shape[0]]

        if detector_backend == "skip":
            face_objs = [(img, img_region, 0)]
        else:
            face_objs = self.detector.detect_face(img, align)


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
        
    
    def get_faces(self,img1_path,img2_path,enforce_detection = True,align = True):
        
        img1_objs = self.extract_faces(img=img1_path, grayscale=False,
                                            enforce_detection=enforce_detection,align=align
        )

        img2_objs = self.extract_faces(img=img2_path, grayscale=False,
                                            enforce_detection=enforce_detection,align=align
        )
        
        
        return img1_objs,img2_objs
        
          
    def __represent(self, img_path, normalization="base"):
        """
        This method extracts faces from an image, normalizes them and returns their embeddings
        """
        # ---------------------------------
        # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
        target_size = (112, 112)

        if isinstance(img_path, str):
            img = load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f"unexpected type for img_path - {type(img_path)}")
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        # --------------------------------
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]
        # ---------------------------------

        resp_objs = []

        for img, region, _ in img_objs:
            img = normalize_input(img=img, normalization=normalization)
            embedding = self.model_obj.predict(img, verbose=0)[0].tolist()
            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_objs.append(resp_obj)

        return resp_objs


    def verify(self,img1_objs,img2_objs, normalization="base"):
        """
        This method extracts faces from two images, computes their embeddings and verifies whether they belong to the same person
        """
        tic = time.time()

        distances = []
        regions = []

        for img1_content, img1_region, _ in img1_objs:
            for img2_content, img2_region, _ in img2_objs:
                img1_embedding_obj = self.__represent(
                    img_path=img1_content,
                    normalization=normalization
                )

                img2_embedding_obj = self.__represent(
                    img_path=img2_content,
                    normalization=normalization
                )

                img1_representation = img1_embedding_obj[0]["embedding"]
                img2_representation = img2_embedding_obj[0]["embedding"]

                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)
                )

                distances.append(distance)
                regions.append((img1_region, img2_region))

        threshold = dst.findThreshold("euclidean_l2")
        distance = min(distances)
        facial_areas = regions[np.argmin(distances)]
        toc = time.time()

        resp_obj = {
            "verified": distance <= threshold,
            "distance": distance,
            "threshold": threshold,
            "detector_backend": "opencv",
            "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
            "Step (3-4) time": round(toc - tic, 2),
        }

        return resp_obj
