import cv2
import numpy as np
import time
import argparse
import os

# Multiprocessing Imports
from multiprocessing import freeze_support
import concurrent.futures
# System Imports
import subprocess
import ctypes
# Control Variables
import config

from utilities import to_ltwh,get_iou,to_ltrd,draw_fancy_bbox
from multi_object_tracking import multitracker
from recognition import face_recognition_dlib
import face_recognition


class secure_access_cv:

    def __init__(self):
        self.m_tracker = multitracker(config.tracker_type)
        self.face_recog = face_recognition_dlib()
        
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        self.futures = None
        self.start_time = None
        
        self.face_identities = []
        
        # Device Action Variables
        self.invoke_screenlock = False
        self.max_wait = 50
        self.lock_iter = 0
        
    def identify(self):
        elasped_time = time.time() - self.start_time
        #print("elasped_time = ",elasped_time)
        if (elasped_time % 1)< 0.2:
            #print("1 sec elasped... Check our executor for result!")
            if not self.futures.running():
                
                try:
                    self.face_identities = self.futures.result(timeout=0.005)
                except concurrent.futures._base.TimeoutError:
                    return

                self.invoke_screenlock = True
                for iter,name in enumerate(self.face_identities):
                    self.m_tracker.Tracked_classes[iter] = name
                    if name !="Unknown":
                        self.m_tracker.colors[iter] = (0,255,0)
                        self.invoke_screenlock = False

    def activate_sa(self,vid_id,dataset_dir =""):
        
        cap = cv2.VideoCapture(vid_id)  # 0 is the index of the default camera
                
        data = self.face_recog.get_embeddings(dataset_dir)
        
        # Adjusting downscale config parameter based on video size.
        # If video frame is large we can downscale more without losing appropriate data
        vid_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        config.downscale = int(vid_width/config.exp_width) * config.downscale 
        
        dth_frame = 10  # process every 5th frame
        frame_count = 0
        upsample_detection = 1
        waitTime = 1
        if config.debug:
            waitTime = 15
            upsample_detection = 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_disp = frame.copy()

            # Containers: Hold detected faces and downsized frame
            face_locations = []
            face_locations_scaled = []
            rgb_small_frame = None
            
            # [Stage 1: Detecting Faces]
            if frame_count % dth_frame == 0:
                small_frame = cv2.resize(frame,(0,0),fx = (1/config.downscale),fy = (1/config.downscale))
                rgb_small_frame = small_frame[:,:,::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame,upsample_detection,config.face_detector)
                # Displaying detected faces
                for face_loc in face_locations:
                    face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,
                                       face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                    face_locations_scaled.append(face_loc_scaled)
                    
                    top,right,bottom,left = face_loc_scaled # css
                    cv2.rectangle(frame_disp,(left,top),(right,bottom),(255,0,0),4)

            # [Stage 2: Tracking -> Initialization]
            if self.m_tracker.mode =="Detection":
                if len(face_locations)!=0:
                    inp_formats = ["css"]*len(face_locations_scaled)
                    bboxes = list(map(to_ltwh,face_locations_scaled,inp_formats))
                    self.m_tracker.track(frame,frame_disp,bboxes)
                    
                    if not self.futures or not self.futures.running():
                        self.futures = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                        self.start_time = time.time()
            else:
                # [Stage 2: Tracking -> Tracking]
                new_bboxes = []
                if len(face_locations)!=0:
                    bboxes_ltrd = list(map(to_ltrd,self.m_tracker.tracked_bboxes))
                    for i, face_loc_scaled in enumerate(face_locations_scaled):
                        face_ltrd = to_ltrd(face_loc_scaled,"css")
                        iou_list = [get_iou(face_ltrd,t_box_ltrd) for t_box_ltrd in bboxes_ltrd]
                        iou = max(iou_list)
                        if iou > 0.5:
                            pass
                        elif iou < 0.2:
                            face_bbox = to_ltwh(face_loc_scaled,"css")
                            new_bboxes.append(face_bbox)
                
                # Detected New Faces? -> Reinitialize Tracker
                if len(new_bboxes)!=0:
                    already_tracked = [np.array(t_bbox) for t_bbox in self.m_tracker.tracked_bboxes]
                    init_bboxes = [*already_tracked,*new_bboxes]
                    self.m_tracker.mode = "Detection"
                    self.m_tracker.track(frame,frame_disp,init_bboxes)
                else:
                    self.m_tracker.track(frame,frame_disp)
                    self.identify()
                    
            if self.invoke_screenlock:
                s=(5,5)
                e=(frame_disp.shape[1]-5,frame_disp.shape[0]-5)
                draw_fancy_bbox(frame_disp,s,e,(0,0,255),4,style ='dotted')
                cv2.putText(frame_disp,f">>> Unauthorized access <<<",(int(frame_disp.shape[1]/2)-155,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(128,0,128),2)
                #cv2.putText(frame_disp, f"Unauthorized access!" ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255))
                cv2.putText(frame_disp, f"Locking in {self.max_wait-self.lock_iter} s" ,(20,60) ,cv2.FONT_HERSHEY_DUPLEX, 1, (128,0,255))
                self.lock_iter = self.lock_iter + 1
                
                r,c = frame_disp.shape[0:2]
                r_s = int(r*0.2)
                r_e = r - r_s
                
                t_elpsd_p = 1 - ((self.max_wait - self.lock_iter)/self.max_wait)
                r_ln = r_e - r_s
                
                cv2.rectangle(frame_disp,(50,r_s),(80,r_e),(0,0,128),2)
                cv2.rectangle(frame_disp,(52,r_s + int(r_ln*t_elpsd_p)),(78,r_e),(0,140,255),-1)
                
                if self.lock_iter== self.max_wait:
                    self.lock_iter = 0
                    self.invoke_screenlock = False
                    ctypes.windll.user32.LockWorkStation()
    

            
            # Display Live Camera Stream
            if config.display or config.debug:
                cv2.imshow('Processed Frame', frame_disp)
                if cv2.waitKey(waitTime) & 0xFF == 27:
                    break
            frame_count += 1
            

        cap.release()
        cv2.destroyAllWindows()







def secure_live():
    # Loading the training and test data
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
    parser.add_argument("-disp","--display", help="Flag to display GUI", type=bool,default=True)
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
    config.display = args.display if not config.display else config.display
    config.debug = args.debug if not config.debug else config.debug
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