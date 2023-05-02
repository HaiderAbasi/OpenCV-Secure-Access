# Warning Controls
ignore_Warning_MMSF = True

experimental = False
nvidia_gpu = False

# Video Path or camera number
vid_id = None

# Debugging controls
display = False
debug = False
display_state = False

# Detection Flags
face_detector = "hog"
cascPathface = r"models\haarcascade_frontalface_alt_tree.xml"

# Tracking Flags
tracker_type = "MOSSE"

# Recognition Flags
exp_width = 640
exp_height = 480
downscale = 2 # Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
new_user = None
authorized_dir = r"authorized_personnels"
recog_tolerance = 0.6