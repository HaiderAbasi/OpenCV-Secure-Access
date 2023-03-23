import cv2
import numpy as np
import time
import argparse
import os

# Multiprocessing Imports
from multiprocessing import freeze_support

# System Imports
import subprocess

# Control Variables
import config



class secure_access_cv:

    def __init__(self):
        pass


    def activate_sa(self,vid_id,dataset_dir =""):
        pass







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