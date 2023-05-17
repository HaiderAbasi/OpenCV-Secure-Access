# Normal/Debug Mode
# # Normal : Secure access on Live/Video
# # Debug  : Secure access on a test video to validate its working
debug = True

# Debugging controls
display = True
display_state = True
write_vid = False
early_stopping = False
verbose = 0

# Access Control
user_to_annoy = "arsal"
user_to_restrict = "taha"

# Beware! Test first on dummy folder about its effects
# Windows
#folder_to_hide = r"D:\Haider\Udemy\Private" # Set this to any folder on your system. 
# Ubuntu
folder_to_hide = r"/home/shadowmaster/Documents/Private" # Set this to any folder on your system. 

# Video Path or camera number
vid_id = None

# Detection Flags
face_detector = "hog"

# Tracking Flags
tracker_type = "MOSSE"

# Recognition Flags
exp_width = 640
exp_height = 480
downscale = 2 # Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
new_user = None
add_user = None
authorized_dir = r"authorized_personnels"
recog_tolerance = 0.6

# Experimental
experimental = False
nvidia_gpu = False

# Warning Controls
ignore_Warning_MMSF = True