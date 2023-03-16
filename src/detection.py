import os
import cv2
import numpy as np
import distance
import math
from PIL import Image


class Detector:
    
    def __init__(self):
        self.detector = {}
        self.detector["face_detector"] = self.build_cascade("haarcascade")
        self.detector["eye_detector"] = self.build_cascade("haarcascade_eye")

    def build_cascade(self, model_name="haarcascade"):
        opencv_path = self.get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
            if os.path.isfile(face_detector_path) != True:
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = opencv_path + "haarcascade_eye.xml"
            if os.path.isfile(eye_detector_path) != True:
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def detect_face(self, img, align=True):
        resp = []

        detected_face = None
        img_region = [0, 0, img.shape[1], img.shape[0]]

        faces = []
        try:
            faces, _, scores = self.detector["face_detector"].detectMultiScale3(
                img, 1.1, 10, outputRejectLevels=True
            )
        except:
            pass

        if len(faces) > 0:

            for (x, y, w, h), confidence in zip(faces, scores):
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

                if align:
                    detected_face = self.align_face(self.detector["eye_detector"], detected_face)

                img_region = [x, y, w, h]

                resp.append((detected_face, img_region, confidence))

        return resp

    def align_face(self, eye_detector, img):
        detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

        eyes = sorted(eyes, key=lambda v: abs((v[0] - v[2]) * (v[1] - v[3])), reverse=True)

        if len(eyes) >= 2:

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            right_eye = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))

            img = self.alignment_procedure(img, left_eye, right_eye)

        return img
    
    @staticmethod
    def alignment_procedure(img, left_eye, right_eye):

        # this function aligns given face in img based on left and right eye coordinates

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        # -----------------------
        # find rotation direction

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # -----------------------
        # find length of triangle edges

        a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

        # -----------------------

        # apply cosine rule

        if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # -----------------------
            # rotate base image

            if direction == -1:
                angle = 90 - angle

            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))

        # -----------------------

        return img  # return img anyway
    
    @staticmethod
    def get_opencv_path():
        opencv_home = cv2.__file__
        folders = opencv_home.split(os.path.sep)[0:-1]

        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        return path + "/data/"


    def __getstate__(self):
        state = self.__dict__.copy()
        del state["detector"]
        return state