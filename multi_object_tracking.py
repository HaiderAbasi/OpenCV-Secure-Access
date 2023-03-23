import cv2
from utilities import putText,get_data,get_fileName,Gui,print_h,draw_fancy_bbox
import time
import config


class multitracker:

    def __init__(self,tracker_type = "CSRT"):
        print(f"\n[Multi-Track]: Created a {tracker_type} based Multi-tracker!\n")
        # Create MultiTracker object        
        self.m_tracker = cv2.MultiTracker_create()        
        self.tracker_type = tracker_type

        self.mode = "Detection"
        self.Tracked_classes = []
        self.colors = []        

        # List of Tracked bboxes for accessibility
        self.tracked_bboxes = [] # [(left,top,width,height)...]
        self.unique_ids = [] # Unique id for each tracked object
        self.obj_iter = 1 # Unique id is generated in order
                          # the track object was initailized

    def tracker_create(self):
        if self.tracker_type == "MOSSE":
            return(cv2.TrackerMOSSE_create())
        elif self.tracker_type == "KCF":
            return(cv2.TrackerKCF_create())
        if self.tracker_type == "CSRT":
            return(cv2.TrackerCSRT_create())

    def track(self,frame,frame_draw,bboxes=[]):
            
        # 4. Checking if SignTrack Class mode is Tracking If yes Proceed
        if(self.mode == "Tracking"):

            # get updated location of objects in subsequent frames
            success, boxes = self.m_tracker.update(frame)

            # draw tracked objects
            Grant_access = False
            if success:
                self.tracked_bboxes = []
                for i, rct in enumerate(boxes):
                    # Appending tracked bboxes to list
                    self.tracked_bboxes.append((round(rct[0],1),round(rct[1],1),round(rct[2],1),round(rct[3],1)))
                    p1 = (int(rct[0]), int(rct[1]))
                    p2 = (int(rct[0] + rct[2]), int(rct[1] + rct[3]))    
                    draw_fancy_bbox(frame_draw, p1, p2, self.colors[i],4, 15, 10)
                    cv2.putText(frame_draw,f"{self.Tracked_classes[i]}",(p1[0],p1[1]-20),cv2.FONT_HERSHEY_DUPLEX,1.0,(128,0,255),2)
                    if self.colors[i] == (0,255,0):
                        Grant_access = True
                        
                if Grant_access and config.display_state:    
                    cv2.putText(frame_draw,f">>> Access granted <<<",(int(frame_draw.shape[1]/2)-120,25),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,128,0),2)
            else:
                self.mode = "Detection"


        # 3. If SignTrack is in Detection Mode -> Proceed to intialize tracker
        elif (self.mode == "Detection"):
            # Incase multitracker initialzied previously -> Delete multitracker object + Create new instance
            if (len(self.m_tracker.getObjects())!= 0):
                del self.m_tracker
                self.m_tracker = cv2.MultiTracker_create()            
                self.colors.clear() # Colors would have also been initialized along with the tracker, Clear it!
                self.Tracked_classes.clear()
                
                # Resetting unique id list to only carry ids of currently tracked objects
                self.unique_ids.clear()


            # Initialize MultiTracker
            for bbox in bboxes:
                if ( (bbox !=[]) and (bbox[2] != 0) and (bbox[3] != 0) ): # Sane bbox (y)
                    tracker = self.tracker_create()
                    self.m_tracker.add(tracker, frame, tuple(bbox))
                    self.mode = "Tracking" # Set mode to tracking
                    self.Tracked_classes.append(f"{self.obj_iter}") # keep tracking frame sign name
                    self.colors.append((0,144,255)) # Orange for (Recognizer not applied yet) detections
                    
                    # Appending tracked object unique id
                    self.unique_ids.append(f"{self.obj_iter}")
                    self.tracked_bboxes = [] # Resetting tracked bboxes
                    # Incrementing obj_iter to move to next tracked object
                    self.obj_iter = self.obj_iter + 1


def demo():
    print_h("[main]: Investigating OpenCV Multi-Tracker module.")
    
    gui = Gui()
    # Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
    # Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
    # Use MOSSE when you need pure speed
    tracker_type = "CSRT"
    m_tracker = multitracker(tracker_type)

    Friends = "Data/NonFree\Friends\Friends_AllClothes.mp4"
    Megamind = "Data/NonFree/Megamind.avi"
    Window_Name = get_fileName(Megamind)

    vid_dirs,filenames = get_data("multitracking")

    #data_iter = int(input("Please select one from the following:\n"+"\n".join(filenames)+"\n"))
    data_iter = gui.selectdata(filenames)
    Window_Name = filenames[data_iter]

    #debugger = debugger(Window_Name,["data_iter"],[len(filenames)])

    #-- 1. Read the video stream
    cap = cv2.VideoCapture(vid_dirs[data_iter])
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)


    while True:
        #debugger.update_variables() # Get updated variables
        #data_iter = debugger.debug_vars[0]

        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        else:
            frame_disp = frame.copy()
            if m_tracker.mode=="Tracking":
                start_time = time.time()
                m_tracker.track(frame,frame_disp)

                # if objects were tracked
                if len(m_tracker.tracked_bboxes)!=0:
                    fps = 1.0 / (time.time() - start_time) if (time.time() - start_time)!=0 else 100.00
                    fps_txt = f"FPS: = {fps:.2f}"
                    putText(frame, f"Tracking ( {tracker_type} ) at {fps_txt}",(20,20))
                else:
                    # else reset mode to detection
                    m_tracker.mode = "Detection" # Reset to Detection
            else:
                putText(frame, f"Detection",(20,20))

            cv2.imshow(Window_Name,frame_disp)
            k = cv2.waitKey(30)
            if k==ord('c'):
                gui.selectROIs(frame)
                bboxes = gui.selected_rois
                # Initliaze Tracker
                m_tracker.track(frame,frame.copy(),bboxes)
            elif k==27:# If Esc Pressed Quit!
                break


if __name__ =="__main__":
    demo()