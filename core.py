import cv2
import numpy as np
import time
import argparse
import os

# Multiprocessing Imports
import concurrent.futures
from multiprocessing import Array,freeze_support,active_children

# System Imports
import subprocess
import ctypes

# Face Recognition Core Imports
from multi_object_tracking import multitracker
from recognition import face_recognition_dlib
import face_recognition # Reference (Cross-Platform-Path) = https://stackoverflow.com/questions/122455/handling-file-paths-cross-platform

# Utility Imports
import config
from collections import deque
from utilities import get_iou,to_ltrd,to_ltwh,draw_fancy_bbox


class secure_access_cv:

    def __init__(self):
        
        # [1: Object Tracker] Loading MultiTracker class for multiple face tracking after detectionh
        self.m_tracker = multitracker(config.tracker_type)
        # [2: Recognition] Loading Face Recognition performing recogntion on detected faces
        self.face_recog = face_recognition_dlib()
        # [3: Profiling] Creating deque object to act as a running average filter for a smoothed FPS estimation
        self.fps_queue = deque(maxlen=10)
        self.elapsed_time = 0
        # [4: Multiprocessing] Creating an object (executor) of the process pool executor in concurrent.futures for multiprocessing
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        self.start_time = None
        self.futures = None
        
        # Array of integer type of length 4 ==> Used here for sharing the computed bbox
        self.face_names_chars = Array('c', 100) 
        # 1: Single, Multiple, Appended
        self.currently_recognizing = "Single"
        
        # Voting System for robust authorization system
        self.voting_dict = {} # Votes for each tracked object      
        self.time_after_recognize = [4,8,16,32,64,128,248] # Wait time before tracked face can be recognized again
        self.tar_counter = {} # index for each tracked face indicating after how many sec we will perform recogniton again
        self.last_recog_time = {} # Last recognition time for each tracked face
        self.perform_recog = {} # perform recognition for selected faces
        self.identifying_faces = False # Performing face identification at this iteration?

        # Device Action Variables
        self.invoke_screenlock = False
        self.max_wait = 50
        self.lock_iter = 0
        
             
    def update_state(self,frame):
        """
        Displays fps the program is run at along with tracker and detector used

        Args:
            frame: The input frame to process.
        """
        elapsed_time_sec = (self.elapsed_time)/1000
        fps = 1.0 / elapsed_time_sec if elapsed_time_sec!=0 else 100.00
        # Rolling average applied to get average fps estimation
        self.fps_queue.append(fps)
        fps = (sum(self.fps_queue)/len(self.fps_queue))
        #fps_txt = f"{self.tracker.mode}: ( {self.tracker.tracker_type} ) at {fps:.2f} FPS"
        fps_txt = f"{fps:.2f} FPS"
        cv2.putText(frame, fps_txt ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
        cv2.putText(frame, f"Detector = {config.face_detector}, Mode = {self.m_tracker.mode}" ,(20,160) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
      
        
    def identify(self):
        """
        function to extract data from a Face recognizer child process, 
        then process it to the desired format, and perform necessary actions
        based on the identities obtained.
        """

        elasped_time = time.time() - self.start_time
        #print("elasped_time = ",elasped_time)
        if (elasped_time % 1)< 0.2:
            #print("1 sec elasped... Check our executor for result!")
            if not self.futures.running():
                
                if self.identifying_faces:
                    print("\n Identifying Faces....")
                    extract_recog_result_start = time.time()
                    
                    try:
                        self.face_names_chars = self.futures.result(timeout=0.005)  # wait up to 1 milli second for the result
                    except concurrent.futures._base.TimeoutError:
                        # executor is busy, skip this task
                        return
                                        
                    extract_recog_result_mini_time = time.time() - extract_recog_result_start
                    print(f"** extract_recog_result_mini_time = {extract_recog_result_mini_time}** ")
                    
                    # 0) Transferring data from a shared array to a list
                    face_names_chars_list = list(self.face_names_chars)
                    # 1) Removing garbadge values
                    if '0' in face_names_chars_list:
                        end_idx = face_names_chars_list.index('0')
                        face_names_chars_list = face_names_chars_list[0:end_idx]
                    # 2) Creating string from the list of chars
                    face_names_str = ''.join(face_names_chars_list)
                    # 3) Seperating face-names on seperater ','
                    face_names = face_names_str.split(",")


                    for iter, name in enumerate(face_names):
                        unique_id = self.m_tracker.unique_ids[iter] # get the unique_id for the tracked object
                        
                        if name != "Unknown":
                            # Authroized individual
                            self.m_tracker.colors[iter] = (0, 255, 0)
                            if unique_id in self.voting_dict: 
                                # if the object has already been tracked, increment its vote count
                                self.voting_dict[unique_id] = self.voting_dict[unique_id] + 1 if self.voting_dict[unique_id] < 3 else self.voting_dict[unique_id] 
                            else: 
                                # if the object is new, add it to the dictionary with a vote count of 0
                                self.voting_dict[unique_id] = 3
                        else:
                            # Un-Authroized individual
                            self.m_tracker.colors[iter] = (0, 0, 255)
                            if unique_id in self.voting_dict: 
                                # if the object has already being tracked, reset its vote count to 0
                                self.voting_dict[unique_id] = self.voting_dict[unique_id] - 1 if self.voting_dict[unique_id] > 0 else self.voting_dict[unique_id] 
                            else:
                                # start with low confidence
                                self.voting_dict[unique_id] = 2
                  
                        # Displaying the face label along with current Votes
                        face_label =  f"{name} ({self.voting_dict[unique_id]})"
                        self.m_tracker.Tracked_classes[iter] = face_label                  
                  
                  
                    for iter, name in enumerate(face_names):
                        unique_id = self.m_tracker.unique_ids[iter] # get the unique_id for the tracked object
                        # Recognition was done on tracked face- We have its unique id.
                        #  We use unique-id to note down its last recogniton time
                        self.last_recog_time[unique_id] = time.time()
                    self.identifying_faces = False
                                              
                    # reset the vote count of all tracked objects that are not present in the current frame
                    keys_to_delete = []
                    for id in self.voting_dict.keys():
                        if id not in self.m_tracker.unique_ids:
                            keys_to_delete.append(id)
                    for id in keys_to_delete:
                        del self.voting_dict[id]
                        if id in self.tar_counter:
                            del self.tar_counter[id]
                            #print("deleting tac counter = ",self.tar_counter)
                        if id in self.last_recog_time:
                            del self.last_recog_time[id]
                            #print("deleting last_recog_time = ",self.last_recog_time)
                        if id in self.perform_recog:
                            del self.perform_recog[id]
                            #print("deleting perform_recog = ",self.perform_recog)

                    # check if all tracked object has received 3 consecutive unauthorized votes
                    screenlock_votes = [vote_count == 0 for vote_count in self.voting_dict.values()]
                    if all(screenlock_votes):
                        self.invoke_screenlock = True # lock the screen if any object has received 3 consecutive unauthorized votes
                        self.lock_iter = 0
                    else:
                        self.invoke_screenlock = False # don't lock the screen if no object has received 3 consecutive unauthorized votes
                    
                    extract_recog_result_time = extract_recog_result_start - time.time()
                    print(f"** extract_recog_result_time = {extract_recog_result_time}** ")
               
                    
    def activate_sa(self,vid_id,dataset_dir =""):   
        """
        Activates secure access by starting face recognition and tracking on the video feed.
        
        Args:
        vid_id: The id of the camera/webcam device to capture frames from or the path of the video file to use.
        dataset_dir: The path to the directory containing the images to use for training the face recognition model.
        
        """
        print("Activating Secure Access....\n Authorized personnel only!")
        config.vid_id = vid_id
        
        # Get embeddings [encodings + Names]
        data = self.face_recog.get_embeddings(dataset_dir)
        # print("data = ",data)
        # cv2.waitKey(0)
        
        if config.ignore_Warning_MMSF or isinstance(vid_id, str):
            # Debugging on a stored video...
            cap = cv2.VideoCapture(vid_id)
        else:
            # Get a reference to webcam #0 (the default one)
            cap = cv2.VideoCapture(vid_id,cv2.CAP_DSHOW)# Suppress Warning_MMSF by adding flag cv2.CAP_DSHOW Reference : https://forum.opencv.org/t/cant-grab-frame-what-happend/4578
        if not cap.isOpened():
            print(f"\n[Error]: Unable to create a VideoCapture object at {vid_id}")
            return
        
        # Adjusting downscale config parameter based on video size. If video frame is large we can downscale more without losing appropriate data
        vid_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        config.downscale = int(vid_width/config.exp_width) * config.downscale 
                
        dth_frame = 15
        upsample_detection = 1
        if config.debug:
            upsample_detection = 2
        
        frame_iter = 0 # Current frame iteration
        while cap.isOpened():
            start_time = time.time() # Get the intial time
            # Grab a single frame of video
            ret, frame = cap.read()
            if not ret:
                print("No-more frames... :(\nExiting!!!")
                break
            frame_draw = frame.copy()
            
            face_locations = [] # (top, right, bottom, left) css
            rgb_small_frame = None
            # Step 1: Detecting face on every dth_frame             
            if frame_iter%dth_frame==0:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=(1/config.downscale), fy=(1/config.downscale))
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                if config.nvidia_gpu:
                    detector = "cnn" # Use CNN (more accurate)
                else:
                    detector = "hog"
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame,upsample_detection,model=detector)
                for face_loc in face_locations:
                    face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,
                                       face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                    top, right, bottom, left = face_loc_scaled
                    cv2.rectangle(frame_draw,(left,top),(right,bottom),(255,0,0),4)


            # Step 2: Start Tracking if faces are present
            if self.m_tracker.mode =="Detection":
                if len(face_locations)!=0: # Found atleast one face
                    if len(face_locations)>1:
                        self.currently_recognizing = "Multiple"
                    else:
                        self.currently_recognizing = "Single"
                       
                    rgb_small_frame, face_locations,face_locations_frame = self.face_recog.preprocess(frame,frame_draw,rgb_small_frame,[],face_locations)
                    
                    # Initialize tracker with bboxes or face_locations (based on detector used) 
                    inp_formats = ["css"]*len(face_locations_frame)
                    faces_hog = list( map(to_ltwh,face_locations_frame,inp_formats) )
                    self.m_tracker.track(frame,frame_draw,faces_hog) # Initialize Tracker

                    
                    if not self.futures or not self.futures.running():
                        # if the previous future has completed or has not started yet, cancel it
                        if self.futures:
                            # Cancel the future object
                            self.futures.cancel()
                            # Delete the future object to free up memory
                            del self.futures
                            
                        # Step 3: Face Recognition using dlib (Child process)
                        # Perform recognition on detected faces as a seperate child process to avoid halting main process
                        self.futures = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                        
                        self.start_time = time.time()
                        self.identifying_faces = True

            else:
                # If detector detected faces other then already tracking. Append them to tracked faces
                new_bboxes = []
                if len(face_locations)!=0: # Found atleast one face
                    # Checking to see if detected faces are being new or not
                    for i, face_loc in enumerate(face_locations):
                        face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                        t_bboxes = list( map(to_ltrd,self.m_tracker.tracked_bboxes) )
                        face_ltrd = to_ltrd(face_loc_scaled,"css")
                        face      = to_ltwh(face_loc_scaled,"css")
                        iou_list = [get_iou(face_ltrd,t_bbox) for t_bbox in t_bboxes]
                        iou = max(iou_list)
                        if iou > 0.5:#Found a match
                            # We have detected a tracked face again.
                            # Check if the time has elapse since its last recognition.
                            #    If yes:
                            #        Performed recognition again.
                            track_iter = iou_list.index(iou)
                            t_id = self.m_tracker.unique_ids[track_iter]
                            if t_id in self.last_recog_time:
                                if t_id not in self.tar_counter:
                                    self.tar_counter[t_id] = 0
                                # We have performed recognition before
                                time_since_recognition = time.time() - self.last_recog_time[t_id]
                                if time_since_recognition >= self.time_after_recognize[self.tar_counter[t_id]]:
                                    # perform face recognition for this face
                                    # Increase the waittime to perform next recognition
                                    self.tar_counter[t_id] = self.tar_counter[t_id] + 1 if self.tar_counter[t_id] < len(self.time_after_recognize) else self.tar_counter[t_id]
                                    self.perform_recog[t_id] = True
                                        
                        elif iou < 0.2:# Detected face not already in tracking list
                            new_bboxes.append(face) 

                if len(new_bboxes)!=0:
                    already_tracked = [np.array(t_bbox) for t_bbox in self.m_tracker.tracked_bboxes]
                    init_bboxes = [*already_tracked,*new_bboxes]
                    self.m_tracker.mode = "Detection"
                    self.m_tracker.track(frame,frame_draw,init_bboxes) # Appending new found bboxes to already tracked

                    if not self.futures or not self.futures.running():
                        # if the previous future has completed or has not started yet, cancel it
                        if self.futures:
                            # Cancel the future object
                            self.futures.cancel()
                            # Delete the future object to free up memory
                            del self.futures
                        # [New evidence added]: If child process is not already executing --> Start recognizer
                        rgb_small_frame, face_locations,_ = self.face_recog.preprocess(frame,frame_draw,rgb_small_frame,init_bboxes)
                        # Perform recognition on detected faces as a seperate child process to avoid halting main process
                        self.futures = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                        self.start_time = time.time()
                        self.identifying_faces = True
                        self.currently_recognizing = "Appended"
                else:
                    # Perform recogntion again if its not already run and a face has not been recognized 
                    # for a long time
                    if ( not self.futures.running() and any(self.perform_recog.values()) ):
                        #print("Performing recognition again.....")
                        recog_while_track_start = time.time()
                        # if the previous future has completed or has not started yet, cancel it
                        if self.futures:
                            # Cancel the future object
                            self.futures.cancel()
                            # Delete the future object to free up memory
                            del self.futures
                        # reset calls:
                        for ids in self.perform_recog.keys():
                            self.perform_recog[ids] = False
                        # # Call recognition captain. And we are assuming we are using hog.
                        #tracked_bboxes = [np.array(t_bbox) for t_bbox in self.m_tracker.tracked_bboxes]
                        # [New evidence added]: If child process is not already executing --> Start recognizer
                        rgb_small_frame, face_locations,_ = self.face_recog.preprocess(frame,frame_draw,rgb_small_frame,[],face_locations)

                        # Perform recognition on detected faces as a seperate child process to avoid halting main process
                        self.futures = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                        
                        self.start_time = time.time()
                        self.identifying_faces = True
                        self.currently_recognizing = "Appended"
                        recog_while_track_time = recog_while_track_start - time.time()
                        print(f"** recog_while_track_time = {recog_while_track_time}** ")
                                                
                    # Check if recognizer has finshed processing?
                    # a) If friendly face present --> Allow access! + Replace (track_id <---> recognized name) and color
                    # b) else                     --> Revoke previliges
                    self.identify()
                    # Track detected faces
                    self.m_tracker.track(frame,frame_draw)
            

            if config.display_state:
                # Updating state to current for displaying...
                self.update_state(frame_draw)
                # get all active child processes
                children = active_children()
                cv2.putText(frame_draw,f"Recognizing {self.currently_recognizing} face/es  with {len(children)} subprocess running!", ( 20,200 ) ,cv2.FONT_HERSHEY_DUPLEX, 1, (128,128,128))
                cv2.putText(frame_draw, f"Processing took  = {self.elapsed_time:.2f} ms" ,(20,120) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,128))


            if self.invoke_screenlock:
                s=(5,5)
                e=(frame_draw.shape[1]-5,frame_draw.shape[0]-5)
                draw_fancy_bbox(frame_draw,s,e,(0,0,255),4,style ='dotted')
                cv2.putText(frame_draw,f">>> Unauthorized access <<<",(int(frame_draw.shape[1]/2)-155,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(128,0,128),2)
                #cv2.putText(frame_draw, f"Unauthorized access!" ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255))
                cv2.putText(frame_draw, f"Locking in {self.max_wait-self.lock_iter} s" ,(20,60) ,cv2.FONT_HERSHEY_DUPLEX, 1, (128,0,255))
                self.lock_iter = self.lock_iter + 1
                
                r,c = frame_draw.shape[0:2]
                r_s = int(r*0.2)
                r_e = r - r_s
                
                t_elpsd_p = 1 - ((self.max_wait - self.lock_iter)/self.max_wait)
                r_ln = r_e - r_s
                
                cv2.rectangle(frame_draw,(50,r_s),(80,r_e),(0,0,128),2)
                
                cv2.rectangle(frame_draw,(52,r_s + int(r_ln*t_elpsd_p)),(78,r_e),(0,140,255),-1)
                
                if self.lock_iter== self.max_wait:
                    self.invoke_screenlock = False
                    ctypes.windll.user32.LockWorkStation()
            
            waitTime = 1
            if config.debug:
                waitTime = 33
            
            if config.display or config.debug:
                cv2.imshow('Video', frame_draw)
                # Hit 'Esc' on the keyboard to quit!
                k = cv2.waitKey(waitTime)            
                if k==27:#Esc pressed
                    break
                
            self.elapsed_time = (time.time() - start_time)*1000 # ms
            frame_iter = frame_iter + 1

        # Release handle to the webcam
        cap.release()
        cv2.destroyAllWindows()


    def __getstate__(self):
        d = self.__dict__.copy()
        # Delete all unpicklable attributes.
        del d['executor']
        del d['future']
        del d['face_names_chars']
        del d['m_tracker']
        return d




def secure_live():
    # Step 1: Loading the training and test data
    secure_acc = secure_access_cv()
    
    secure_acc.activate_sa(0,dataset_dir= config.authorized_dir)

def demo():
    # Step 1: Loading the training and test data
    secure_acc = secure_access_cv()

    test_vid_path = r"data\test\classroom.mp4"
    dataset_dir = r"data\test\known"

    secure_acc.activate_sa(os.path.abspath(test_vid_path),dataset_dir= os.path.abspath(dataset_dir))


if __name__== "__main__":
    # Pyinstaller fix (Reference : https://stackoverflow.com/a/32677108/11432131)
    freeze_support()

    parser = argparse.ArgumentParser("--Secure Access--")
    parser.add_argument("-disp","--display", help="Flag to display GUI", action= "store_true" )
    parser.add_argument("-d","--debug", help="Debug App", action= "store_true" )
    parser.add_argument("-exp","--experimental", help="Turn Experimental features On/OFF", type=bool,default=False)
    parser.add_argument("-w","--ignore_warnings", help="Ignore warning or not", type=bool,default=True)
    parser.add_argument("-s","--display_state", help="display current (app) state",  action= "store_true")
    parser.add_argument("-fd","--face_detector", help="type of detector for face detection", type=str,default="hog")
    parser.add_argument("-ft","--tracker_type", help="type of tracker for face tracking", type=str,default="MOSSE")
    parser.add_argument("-ds","--downscale", help="downscale amount for faster recognition", type=int,default=2)# Default = 2 -> Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
    parser.add_argument("-rt","--recog_tolerance", help="Adjust tolerance for recognition e.g: (Strict < 0.5 < Loose) ", type=float,default=0.6)# Default = 2 -> Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
    parser.add_argument("-nu","--new_user", help="Add new user to list of authorized personnels", type=str,default=None)
    args = parser.parse_args()
    
    # Set config variables based on parsed arguments
    config.display = args.display
    config.debug = args.debug
    config.ignore_Warning_MMSF = args.ignore_warnings
    config.display_state = args.display_state
    config.face_detector = args.face_detector
    config.tracker_type = args.tracker_type
    config.downscale = args.downscale
    config.recog_tolerance = args.recog_tolerance
    config.new_user = args.new_user
    config.experimental = args.experimental
    
    if config.experimental:
        # Experimental: Features under rigorous testing.
        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected!')
            config.nvidia_gpu = True
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            print('No Nvidia GPU in system!')

    if config.debug:
        demo()
    else:
        secure_live()