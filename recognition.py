import cv2
import numpy as np
import os
import pickle
import face_recognition
import sys

from utilities import to_trbl,to_ltwh,list_images,putText

import time
from time import sleep
import config

class face_recognition_dlib():

    def __init__(self):
        #self.default_embeddings_path = os.path.join(os.getcwd(),"models/live_embeddings-face_enc")
        self.default_embeddings_path = os.path.join(os.path.dirname(__file__),"models/live_embeddings-face_enc")
        self.preprocess_resize = 4 # Downscaling 4 times to improve computation time


    @staticmethod
    def __create_user_dataset(user_dataset_dir):
        
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

            msg = "NOTE: Press (Enter) when ready to grab photos..!"
            putText(frame_draw,msg,(20,20))
            cv2.imshow("New User Authorization",frame_draw)
            k = cv2.waitKey(1)
            if k==13:# Enter key
                saving_user_face = True
            elif k==27: # Esc key
                print("Exit key pressed...\nClosing App!")
                sys.exit(0)

    def __add_new_user(self,user_dataset_dir,write_mode = "wb"):
        image_dirs = list(list_images(os.path.join(os.path.dirname(__file__),user_dataset_dir)))
        
        if not os.path.isdir(user_dataset_dir):
            print(f"\nCreated an empty directory at {user_dataset_dir}\n")
            #os.mkdir(user_dataset_dir)
            os.makedirs(user_dataset_dir)
            print(f"\n1) Grab few photos of {config.new_user} live from camera feed..")
            print("\nNote: Press (Enter) when ready to grab photos..!")
            self.__create_user_dataset(user_dataset_dir)
        #elif (list(paths.list_images(os.path.join(os.path.dirname(__file__),user_dataset_dir)))) == []:
        elif image_dirs == []:
            print(f"\n[Error]: No training-images found at {user_dataset_dir}")
            print(f"\n1) Grab few photos of {config.new_user} live from camera feed..")
            print("\nNote: Press (Enter) when ready to grab photos..!")
            self.__create_user_dataset(user_dataset_dir)
        else:
            print(f"Dataset already available at {user_dataset_dir}\n")
        print(f"Generating {config.new_user} embeddings and appending to authorized personnel list.")
        self.__generate_embeddings(user_dataset_dir,False,opentxtmode=write_mode)

    def __generate_embeddings(self,data_dir,make_default = False,opentxtmode = "wb"):

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
                if len(os.listdir(dataset_dir)) != 0:
                    print(f"Generating embeddings from given dataset at {dataset_dir}")
                    self.__generate_embeddings(dataset_dir,True)
                else:
                    print("\n--------------------------------------------------------------------------------------")
                    print(f"\nLooks like a first run :)...\nWe will add you as an authorized personnel now!\n")
                    name = input('====================== Enter your name ======================\n') 
                    if name!="":
                        user_dataset_dir = os.path.join(os.path.dirname(__file__),os.path.join(config.authorized_dir,name))
                        self.__add_new_user(user_dataset_dir,"wb")
                    else:
                        print("No name provided!\nExiting!!!")
                        sys.exit()
            else:
                print(f"Embeddings already available at {embeddings_path}")
                if config.new_user is not None:
                    print(f"\nAdding new user {config.new_user} to list of authorized personnel")
                    user_dataset_dir = os.path.join(os.path.dirname(__file__),os.path.join(config.authorized_dir,config.new_user))
                    # Add new user to list of authorized personnel
                    self.__add_new_user(user_dataset_dir,"wb")
                elif config.add_user is not None:
                    print(f"\nAdding new user {config.add_user} to list of authorized personnel")
                    user_dataset_dir = os.path.join(os.path.dirname(__file__),os.path.join(config.authorized_dir,config.add_user))
                    # Add new user to list of authorized personnel
                    self.__add_new_user(user_dataset_dir,"ab")
         
                    
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
                
        if bboxes != []:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=(1/config.downscale), fy=(1/config.downscale))
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            scale = config.downscale
            scales = [1/scale]*len(bboxes)
            if len(bboxes)==0:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
            else:
                # Convert found bboxes (ltwh) ===> face locations (ltrd)
                face_locations = list( map(to_trbl,bboxes,scales) )
        else:
            if rgb_small_frame is not None:
                # It has been resized previously for hog face detector 
                scale = int(frame_draw.shape[0]/rgb_small_frame.shape[0]) # Evaluated the resize amount
            else:
                scale = config.downscale
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=(1/scale), fy=(1/scale))
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

            #print(f"frame downscaled {scale} times ")

        # Display the detections
        face_locations_frame = []
        face_names = list(range(len(face_locations)))
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            # Draw a box around the face
            cv2.rectangle(frame_draw, (left, top), (right, bottom), (255,0,0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame_draw, (left, bottom - (9*scale)), (right, bottom), (255,0,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_draw, str(name), (left + 6, bottom - (2*scale)), font, 0.25*scale, (255, 255, 255), 1)
            face_locations_frame.append((top, right, bottom, left))

        return rgb_small_frame,face_locations,face_locations_frame
    
    def recognize(self,rgb_small_frame,face_locations,data):
        # Step 3: Landmark Extraction + Face alignment + Encoding Generation
        # face alignment is part of the process of generating face encodings
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_identities = []
        
        for face_encoding in face_encodings:
            # Step 4: Compare encodings with encodings in data["encodings"]
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)

            faces_matched = face_distances < config.recog_tolerance
            # Create a dictionary to store the number of matches for each person
            match_counts = {}
            match_distances = {}
            # Initialize the match_counts dictionary with a value of 0 for each name
            #match_counts = {name: 0 for name in set(data["names"])}
            #match_distances = {name: 0 for name in set(data["names"])}
            for d_name in data["names"]:
                if d_name not in match_counts.keys():
                    match_counts[d_name] = 0
                    match_distances[d_name] = 0

            # Convert the numpy array to a list and count the number of matches for each name
            for idx,is_matched in enumerate(faces_matched):
                match_counts[data["names"][idx]] = match_counts[data["names"][idx]] + 1 if is_matched else match_counts[data["names"][idx]]
                match_distances[data["names"][idx]] = match_distances[data["names"][idx]] + face_distances[idx] if is_matched else match_distances[data["names"][idx]]

            if config.verbose:
                print("\nmatched_counts = ",match_counts)
                
            match_counts = {key: value for key, value in match_counts.items() if value != 0}
            match_distances = {key: value for key, value in match_distances.items() if value != 0}


            if match_counts and len(match_counts)<3:
                # Find the person with the most matches
                max_matches = max(match_counts.values())
                max_match_names = [name_ for name_, count in match_counts.items() if count == max_matches]
                
                if config.verbose:
                    print("faces_matched = ",faces_matched)
                    print("max_match_names = ",max_match_names)
                
                # Set confident flag based on the number of people with the most matches
                if len(max_match_names) == 1:
                    name = max_match_names[0]
                elif len(max_match_names) == 2:
                    per_a = max_match_names[0]
                    per_b = max_match_names[1]
                    
                    if config.verbose:
                        print(f"per_a = {per_a} - per_b = {per_b}")
                    
                    #max_allowed = (match_counts[per_a] * config.recog_tolerance)
                    min_allowed_diff = match_counts[per_a]*0.05
                    difference = match_distances[per_a] - match_distances[per_b]
                    if config.verbose:
                        print("difference = ",difference)
                    if abs(difference) > min_allowed_diff:
                        if difference>0:
                            name = per_b
                        else:
                            name = per_a
                
            face_identities.append(name)
            if config.verbose:
                print(face_distances ,"\n predicted = ",name)

        return face_identities