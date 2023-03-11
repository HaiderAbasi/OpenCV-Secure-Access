import time
import distance as dst
import ArcFace
import numpy as np
import cv2

from functions import initialize_folder,extract_faces,load_image,normalize_input

class FaceRecognizer:
    
    def __init__(self):
        # Intiialzing home folder for keeping arcface model and weights
        initialize_folder()
        
        # Loading arcface model
        self.model_obj = ArcFace.loadModel()
    


    def represent(self, img_path, enforce_detection=True, detector_backend="opencv", align=True, normalization="base"):
        """
        This method extracts faces from an image, normalizes them and returns their embeddings
        """
        # ---------------------------------
        # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
        target_size = (112, 112)
        if detector_backend != "skip":
            img_objs = extract_faces(
                img=img_path,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )
        else:  # skip
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


    def verify(self, img1_path, img2_path, enforce_detection=True, align=True, normalization="base"):
        """
        This method extracts faces from two images, computes their embeddings and verifies whether they belong to the same person
        """
        tic = time.time()

        img1_objs = extract_faces(img=img1_path, grayscale=False,
                                            enforce_detection=enforce_detection,align=align
        )

        img2_objs = extract_faces(img=img2_path, grayscale=False,
                                            enforce_detection=enforce_detection,align=align
        )

        distances = []
        regions = []

        for img1_content, img1_region, _ in img1_objs:
            for img2_content, img2_region, _ in img2_objs:
                img1_embedding_obj = self.represent(
                    img_path=img1_content,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization
                )

                img2_embedding_obj = self.represent(
                    img_path=img2_content,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
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
            "time": round(toc - tic, 2),
        }

        return resp_obj
