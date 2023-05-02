import cv2
import numpy as np
import os
import pickle
import face_recognition
import sys

from utilities import to_trbl,to_ltwh,list_images

from time import sleep
import config

class face_recognition_dlib():

    def __init__(self):
        #self.default_embeddings_path = os.path.join(os.getcwd(),"models/live_embeddings-face_enc")
        self.default_embeddings_path = os.path.join(os.path.dirname(__file__),"models/live_embeddings-face_enc")
        self.preprocess_resize = 4 # Downscaling 4 times to improve computation time

    @staticmethod
    def create_user_dataset(user_dataset_dir):
        # [1: Face Detector] Loading cascade classifier as the fastest face detector in OpenCV
        faceCascade = cv2.CascadeClassifier()
        if not faceCascade.load(os.path.join(os.path.dirname(__file__),config.cascPathface)):
            print(f'--(!)Error loading face cascade from {config.cascPathface}')
            sys.exit(0)
        if config.ignore_Warning_MMSF or isinstance(config.vid_id, str):
            # Debugging on a stored video...
            cap = cv2.VideoCapture(config.vid_id)
        else:
            # Get a reference to webcam #0 (the default one)
            cap = cv2.VideoCapture(config.vid_id,cv2.CAP_DSHOW)# Suppress Warning_MMSF by adding flag cv2.CAP_DSHOW Reference : https://forum.opencv.org/t/cant-grab-frame-what-happend/4578
        if not cap.isOpened():
            print(f"\n(get_faces)[Error]: Unable to create a VideoCapture object at {config.vid_id}")
            return
        
        if config.face_detector == "hog":
            dth_frame = 15
        else:
            dth_frame = 4
            
        frame_iter = 0 # Current frame iteration
        saved_face_iter = 0
        max_faces = 4
        
        saving_user_face = False
        saved_dataset = False
        while ( (cap.isOpened()) and (not saved_dataset) ):
            # Grab a single frame of video
            ret, frame = cap.read()
            if not ret:
                print("No-more frames... :(\nExiting!!!")
                break
            frame_draw = frame.copy()
            
            faces = [] # (x,y,w,h) bbox format
            
            rgb_small_frame = None
            # Step 1: Detecting face on every dth_frame             
            if frame_iter%dth_frame==0:
                if config.face_detector=="hog":
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=(1/config.downscale), fy=(1/config.downscale))
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    for face_loc in face_locations:
                        face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                        faces.append(to_ltwh(face_loc_scaled,"css"))
                        #top, right, bottom, left = face_loc_scaled
                        #cv2.rectangle(frame_draw,(left,top),(right,bottom),(0,255,0),2)
                else:
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray)          
                
                if saving_user_face:
                    color = (0,255,0)
                else:
                    color = (0,215,255)
                # Displaying detected face in image
                for face in faces:
                    x,y,w,h = face
                    cv2.rectangle(frame_draw,(x,y),(x+w,y+h),color,2)
                    
            if saving_user_face: 
                if len(faces)==1:
                    cv2.putText(frame_draw, "Gathering dataset..." ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
                    #new_user_face = frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
                    img_path = os.path.join(f"{user_dataset_dir}",f"{saved_face_iter}.png")
                    print(f"\nSaving {config.new_user} to authorized personnel dataset at {img_path}")
                    cv2.imwrite(img_path,frame)
                    saved_face_iter = saved_face_iter + 1
                    if saved_face_iter>=max_faces:
                        print(f"Required data saved at {user_dataset_dir}...exiting loop!")
                        saved_dataset = True
                elif len(faces)>1:
                    cv2.putText(frame_draw, "Keep only {config.new_user} in front of the camera!" ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255))

            cv2.imshow("New User Authorization",frame_draw)
            k = cv2.waitKey(1)
            if k==13:# Enter key
                saving_user_face = True
            elif k==27: # Esc key
                print("Exit key pressed...\nClosing App!")
                sys.exit(0)

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
        embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
        #get paths of each file in folder named Images
        #Images here contains my data(folders of various persons) i.e. Friends/..
        imagePaths = list(list_images(os.path.join(os.path.dirname(__file__),data_dir)))
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
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb,model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
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
            sleep(1)
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
            embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
            if not os.path.isfile(embeddings_path): # Only generate if not already present
                print(f"Generating embeddings from given dataset at {dataset_dir}")
                self.generate_embeddings(dataset_dir,True)
            else:
                print(f"Embeddings already available at {embeddings_path}")
                if config.new_user is not None:
                    print(f"\nAdding new user {config.new_user} to list of authorized personnel")
                    user_dataset_dir = os.path.join(os.path.dirname(__file__),os.path.join(config.authorized_dir,config.new_user))
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

    def preprocess(self,frame,frame_draw,rgb_small_frame = None,bboxes = [],face_locations = []):
                
        # Write Code here...
        pass
    
    def recognize(self,rgb_small_frame,face_locations,data):
        
        # Step 2 & 3 [Alignment + Representation]
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_identities = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"],face_encoding,config.recog_tolerance)
            
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(data["encodings"],face_encoding)
            min_dist_idx = np.argmin(face_distances)
            
            if matches[min_dist_idx]:
                name = data["names"][min_dist_idx]
                
            face_identities.append(name)
            
        
        return face_identities