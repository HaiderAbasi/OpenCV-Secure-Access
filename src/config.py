# Warning Controls
ignore_Warning_MMSF = True

repo_dir = ""

experimental = False
nvidia_gpu = False

vid_id = None

# Debugging controls
debug = True
display_state = False

# Detection Flags


# Tracking Flags
tracker_type = "MOSSE"

# Recognition Flags
exp_width = 640
exp_height = 480
downscale = 2 # Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
new_user = None
authorized_dir = r"data/authorized"

recog_tolerance = 0.6

#FIXME:
# 4) Similar faces comparison needs further testing NOTE: Investigating setting a stricter threshold and analyze results
# 5) HOG detector struggles in Low light-conditions NOTE: Test experimental features of CNN-Network for Facedetection in case of GPU

#TODO:
# Downscale =2 seems adequate both in terms of detecting faces at a reasonable distance also with good speed
# 1) Test Seperate downscalle for detection and seperate for recognition xD. See if recognition improves/suffers NOTE: Done at 2
# 2) Test lower different detection frames for detection and see if detection improves # NOTE: Lowering detection frames to 15 seems to improve face detection by more face detected then less + FPS seems okay. Needs further testing on Live
# 3) Investigate in some cases detection happens but tracking doesnot start # NOTE: Tracking does start but it seems to fail and has no connection if its first detection or appended

# 4) Check Facial recogniion on similar faces + tests alternating threshold for a match # NOTE: 
# 5) Test low light conditions and see the results # NOTE: low light conditions affect majorly for hog but extremely cascade
#                                                          If Gpu Available: Use CNN