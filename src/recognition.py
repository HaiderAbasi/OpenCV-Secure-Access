import time
import distance as dst
import numpy as np
import cv2
import os
import sys
import pickle

import config
from detection import Detector
import ArcFace
from functions import initialize_folder,load_image,normalize_input,list_images



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
        self.default_embeddings_path = os.path.join(config.repo_dir,"models/live_embeddings-face_enc")

        # !) Creating Detector object for face detecion and alignment
        self.detector = Detector()

        # 2) Loading Arcface model for face representation
        self.model_obj = ArcFace.loadModel()
        
    def add_new_user(self,user_dataset_dir):
        image_dirs = list(list_images(os.path.join(os.path.dirname(__file__),user_dataset_dir)))
        if not os.path.isdir(user_dataset_dir):
            print(f"\nCreated an empty directory at {user_dataset_dir}\n")
            os.mkdir(user_dataset_dir)
            print(f"\n1) Grab few photos of {config.new_user} live from camera feed..")
            print("\nNote: Press (Enter) when ready to grab photos..!")
            self.create_user_dataset(user_dataset_dir)
        #elif (list(paths.list_images(os.path.join(os.path.dirname(__file__),user_dataset_dir)))) == []:
        elif image_dirs == []:
            print(f"\n[Error]: No training-images found at {user_dataset_dir}")
            print(f"\n1) Grab few photos of {config.new_user} live from camera feed..")
            print("\nNote: Press (Enter) when ready to grab photos..!")
            self.create_user_dataset(user_dataset_dir)
        else:
            print(f"Dataset already available at {user_dataset_dir}\n")
        print(f"Generating {config.new_user} embeddings and appending to authorized personnel list.")
        self.generate_embeddings(user_dataset_dir,True,opentxtmode="ab")
    
    
    def generate_embeddings(self,data_dir,make_default = False,opentxtmode = "wb"):

        if config.debug:
            foldername = os.path.basename(data_dir)
        else:
            foldername = os.path.basename(config.authorized_dir)
        embeddings_path = os.path.join(config.repo_dir,f"models/{foldername}-live_embeddings-face_enc")
        #get paths of each file in folder named Images
        #Images here contains my data(folders of various persons) i.e. Friends/..
        imagePaths = list(list_images(os.path.join(config.repo_dir,data_dir)))
        if imagePaths == []:
            print(f"\n[Error]: No training-images found at {data_dir}")
            sys.exit(0)
        knownEncodings = []
        knownNames = []
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path (folder name)
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)

            # Step 1 & 2 (Retriveiving face objects)
            face_objs = self.extract_faces(img=image, grayscale=False, enforce_detection=True, align=True)
            face_content = face_objs[0][0] # only one face is present and that is in the first index
            # Step 3: Compute the facial embedding for the face
            encoding = ( self.__represent(img_path=face_content, normalization="base") )[0]["embedding"]
            # loop over the encodings
            # Save the encoding
            knownEncodings.append(encoding)
            knownNames.append(name)
        #save emcodings along with their names in dictionary data
        data = {"encodings": knownEncodings, "names": knownNames}
        
        if opentxtmode == "ab":
            #print("Appending Data = ",data,f" to {embeddings_path}")
            # load the known faces and embeddings saved in last file
            orig_data = pickle.loads(open(embeddings_path, "rb").read())
            data["encodings"] = orig_data["encodings"] + data["encodings"]
            data["names"]     = orig_data["names"]     + data["names"]
        
        #use pickle to save data into a file named after the data folder for later use
        f = open(embeddings_path,"wb")
        f.write(pickle.dumps(data))
        f.close()
        
        if make_default:
            time.sleep(1)
            print(f"\nSetting current embedding at {embeddings_path} as default({self.default_embeddings_path}).")
            #use pickle to save data into a file named after the data folder for later use
            f = open(self.default_embeddings_path,"wb")
            f.write(pickle.dumps(data))
            f.close()


    def get_embeddings(self,dataset_dir="",embeddings_path=""):
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif dataset_dir!="":
            foldername = os.path.basename(dataset_dir)
            embeddings_path = os.path.join(config.repo_dir,f"models/{foldername}-live_embeddings-face_enc")
            if not os.path.isfile(embeddings_path): # Only generate if not already present
                print(f"Generating embeddings from given dataset at {dataset_dir}")
                self.generate_embeddings(dataset_dir,True)
            else:
                print(f"Embeddings already available at {embeddings_path}")
                if config.new_user is not None:
                    print(f"\nAdding new user {config.new_user} to list of authorized personnel")
                    user_dataset_dir = os.path.join(config.repo_dir,os.path.join(config.authorized_dir,config.new_user))
                    # Add new user to list of authorized personnel
                    self.add_new_user(user_dataset_dir)
                    
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
            #print("Loading Data = ",data,f" from {embeddings_path}")
            #data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        elif os.path.isfile(self.default_embeddings_path):
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        else:
            print(f"Face-embedddings {self.default_embeddings_path} is not available.\nProvide dataset to generate new embeddings...")
            exit(0)
        return data

    
    # Face Recognition Pipeline: Step 1 & Step 2 (Detection & Alignment)
    def extract_faces(self, img, grayscale=False,
                            enforce_detection=True, align=True):
        
        target_size = (112, 112)

        # this is going to store a list of img itself (numpy), it region and confidence
        extracted_faces = []

        # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
        img = load_image(img)
        img_region = [0, 0, img.shape[1], img.shape[0]]

        face_objs = self.detector.detect_face(img, align)

        # in case of no face found
        if len(face_objs) == 0 and enforce_detection is True:
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

    def preprocess(self,image,bboxes):
        # this is going to store a list of img itself (numpy), it region and confidence
        extracted_faces = []
        
        for bbox in bboxes:
                x,y,w,h = bbox
                img_pixels = image[y:y+h,x:x+w]
                
                # int cast is for the exception - object of type 'float32' is not JSON serializable
                region_obj = {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "w": int(bbox[2]),
                    "h": int(bbox[3]),
                }
                confidence = 1.0

                extracted_face = [img_pixels, region_obj, confidence]
                extracted_faces.append(extracted_face)
                
        return extracted_faces
        
        
        
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


    def verify(self,data,test_objs, normalization="base"):
        """
        This method extracts faces from two images, computes their embeddings and verifies whether they belong to the same person
        """
        tic = time.time()

        distances = []
        # initialize variables to keep track of the minimum distance and the corresponding image content
        min_distance = float('inf')
        min_img_content = None
        for poi_representation in data['encodings']:
            for img2_content, _ , _ in test_objs:

                img2_embedding_obj = self.__represent(img_path=img2_content,normalization=normalization)
                img2_representation = img2_embedding_obj[0]["embedding"]

                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(poi_representation), 
                    dst.l2_normalize(img2_representation)
                )

                distances.append(distance)
                if distance < min_distance:
                    # update the minimum distance and corresponding image content
                    min_distance = distance
                    min_img_content = img2_content
                    
                    
        threshold = dst.findThreshold("euclidean_l2")
        distance = min(distances)

        # if distance<=threshold:
        #     print("distances = ",distances)
        #     print(f"minimum distance {distance} is found at index {distances.index(distance)}")

        toc = time.time()
        # denormalize the image pixels
        denorm_pixels = min_img_content * 255
        denorm_pixels = np.clip(denorm_pixels, 0, 255)  # clip values to [0, 255]
        denorm_pixels = denorm_pixels.astype(np.uint8)  # convert to uint8 dtype

        resp_obj = {
            "verified": distance <= threshold,
            "img_content": denorm_pixels[0],
            "distance": distance,
            "threshold": threshold,
            "detector_backend": "opencv",
            "Step (3-4) time": round(toc - tic, 2),
        }

        return resp_obj
    
    
    def recognize(self,image,data,faces):
        
        face_names_chars = []
        threshold = dst.findThreshold("euclidean_l2")
        
        for x,y,w,h in faces:
            face_img = image[y:y+h,x:x+w]
            # Step 1 & 2 (Retriveiving face objects)
            face_objs = self.extract_faces(img=face_img, grayscale=False, enforce_detection=False, align=True)
            face_content = face_objs[0][0] # only one face is present and that is in the first index
            # Step 3: Compute the facial embedding for the face
            test_representation = ( self.__represent(img_path=face_content, normalization="base") )[0]["embedding"]
            
            distances = []
            for poi_representation in data['encodings']:
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(poi_representation), 
                    dst.l2_normalize(test_representation))

                distances.append(distance)

            distance = min(distances)
                        
            name = "Unknown"
            if distance <= threshold:
                # Face matched with one of the authorized personel
                #  a) Name the authorized personnel
                #  b) keep its bbox to display
                min_dist_index = distances.index(distance)
                name = data['names'][min_dist_index]
                
            if len(face_names_chars)!=0:
                face_names_chars.append(',')
            face_names_chars = face_names_chars + list(name)
            
        if (len(face_names_chars)>=100):
            face_names_chars = face_names_chars[0:100]
        else:
            lst = ['0'] * (100-len(face_names_chars))
            face_names_chars = face_names_chars + lst
            
        return face_names_chars