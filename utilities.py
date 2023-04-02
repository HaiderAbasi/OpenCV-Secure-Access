import cv2
import os
import math
import numpy as np

def print_h(str):
    print(f"\n##############################################\n{str}")

def put_Text(
    img,
    text,
    uv_top_left,
    bg_color=(255, 255, 255),
    text_color=(255, 255, 255),
    fontScale=0.6,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0,0,0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline and a background slightly larger than the text.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    # Calculate the size of the text box
    text_size = np.zeros((2,))
    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        text_size[0] = max(text_size[0], w)
        text_size[1] += h * line_spacing

    # Calculate the top-left and bottom-right coordinates of the background box
    bg_size = text_size + np.array([10, 10])  # add some padding around the text
    bg_top_left = uv_top_left - [5, 5]
    bg_bottom_right = bg_top_left + bg_size

    # Draw the background box
    draw_fancy_bbox(
        img=img,
        pt1=tuple(bg_top_left.astype(int)),
        pt2=tuple(bg_bottom_right.astype(int)),
        color=bg_color,
        thickness=2,
    )

    # Draw the text
    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=text_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]



def putText(img, text,org=(0, 0),font=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0, 255, 0),thickness=1,color_bg=(0, 0, 0)):
    FONT_SCALE = 2.5e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images

    if fontScale==1:
        height, width= img.shape[:2]
        fontScale = min(width, height) * FONT_SCALE
        thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

    x, y = org
    rect_x = x ; rect_y = y
    ext = 10
    if x > ext:
        rect_x = rect_x - ext
    if y > ext:
        rect_y = rect_y - ext

    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_w, text_h = text_size
    rect_w = text_w
    rect_h = text_h
    if img.shape[1] - (rect_x + text_w) > ext:
        rect_w = rect_w + ext
    if img.shape[0] - (rect_y + text_h) > ext:
        rect_h = rect_h + ext

    #cv2.rectangle(img, org, (x + text_w, y + text_h), color_bg, -1)
    cv2.rectangle(img, org, (rect_x + rect_w, rect_y + rect_h), color_bg, -1)
    cv2.putText(img, text, (x,int( y + text_h + fontScale - 1)), font, fontScale, color, thickness)



class Gui():

    def __init__(self):
        self.pt = None # point instance variable

        # ADVANCED #
        self.img_draw = None
        self.ix,self.iy = -1,-1
        self.fx,self.fy = -1,-1
        self.roi_confirmed = False
        self.selected_rois = []


    # mouse callback function
    def __selectroi_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix,self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_confirmed:
                cv2.rectangle(self.img_draw,(self.ix,self.iy),(self.fx,self.fy),(0,255,0),2)
                if self.ix <= self.fx :
                    strt_col = self.ix
                    width = self.fx - self.ix
                else:
                    strt_col = self.fx
                    width = self.ix - self.fx
                if self.iy <= self.fy:
                    strt_row = self.iy
                    height = self.fy - self.iy
                else:
                    strt_row = self.fy
                    height = self.iy - self.fy
                self.selected_rois.append((strt_col,strt_row,width,height))
                self.roi_confirmed = False
                
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(self.img_draw,(self.ix,self.iy),(x,y),(0,140,255),2)
            self.fx = x
            self.fy = y

    def selectROIs(self,img,title = 'SelectROIs'):
        self.img_draw = img.copy()# Dont want to mess up the original XD
        cv2.namedWindow(title)
        cv2.setMouseCallback(title,self.__selectroi_callback)

        while(1):
            cv2.imshow(title,self.img_draw)
            k = cv2.waitKey(1) & 0xFF
            if k == 13:# Enter
                self.roi_confirmed = True
            elif k == 27:
                break
        cv2.destroyAllWindows()
        #print("selected_rois = ",self.selected_rois)
    
    @staticmethod
    def __nothing(val):
        pass

    def selectdata(self,filenames,Window_Name ="Data Selection",trackbar_name = "choice",useMouse = False,onTop =False,data_type = "test data"):
        
        user_choice = 0

        curr_x = 0
        curr_y = 0
        data_selected = False
        def onMouse(event,x,y,flags,param): # Inner function
            nonlocal curr_x,curr_y,data_selected
            if event == cv2.EVENT_LBUTTONDOWN:
                curr_x,curr_y = x,y
            elif event == cv2.EVENT_LBUTTONUP:
                data_selected = True

        def getUserChoice():
            nonlocal unit_row,shift,curr_y
            #curr_y = unit_row+(shift*user_choice)
            user_choice = ( curr_y - unit_row ) / shift
            return user_choice

        cv2.namedWindow(Window_Name)
        if onTop:
            cv2.setWindowProperty(Window_Name, cv2.WND_PROP_TOPMOST, 1) # Reference: https://stackoverflow.com/a/66364178/11432131
        hp_col = 700
        filename_w = 25
        hp_row = 40 + filename_w*(len(filenames))

        home_page = np.zeros((hp_row,hp_col,3),np.uint8)#BGR
        txt_to_display = f"Please choose one of the following as the {data_type}...."
        cv2.putText(home_page,txt_to_display,(20,20),cv2.FONT_HERSHEY_PLAIN,1.2,(0,140,255),1)
        
        col = 20
        unit_row = 50
        shift = 25
        for idx,filename in enumerate(filenames):
            txt_to_display = f"{idx}: {filename}" 
            cv2.putText(home_page,txt_to_display,(col,unit_row+(shift*idx)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
        cv2.createTrackbar(trackbar_name,Window_Name,user_choice,len(filenames)-1,self.__nothing)

        cv2.setMouseCallback(Window_Name,onMouse)


        prev_txt_to_display = ""
        prev_user_choice = 0
        while(1):
            if useMouse:
                choice = getUserChoice()
                if choice>0:
                    user_choice = round(choice)
                else:
                    user_choice = 0 # If it is less then zero or not choose. Consider it default : 0
                if data_selected:
                    print(">>>>> User_choice = ",user_choice)
                    cv2.waitKey(0)
            else:
                user_choice = cv2.getTrackbarPos(trackbar_name,Window_Name)

            txt_to_display = f"{user_choice}: {filenames[user_choice]}" 
            cv2.putText(home_page,prev_txt_to_display,(col,unit_row+(shift*prev_user_choice)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
            cv2.putText(home_page,txt_to_display,(col,unit_row+(shift*user_choice)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            prev_txt_to_display = txt_to_display
            prev_user_choice = user_choice

            cv2.imshow(Window_Name,home_page)
            k = cv2.waitKey(1) & 0xFF
            if ( (k == 13) or data_selected ):# Enter or data choosen by mouses
                if not data_selected:
                    cv2.destroyWindow(Window_Name)# No longer need it       
                break # Exit loop
            elif k==27: #Esc pressed -> Exiting...
                user_choice = -1
                break

        return user_choice

    # ADVANCED #



    def ret_point(self,event,x,y, flags, param):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            self.pt = (x,y)

    def select_cnt(self,img,cnts,Loop=False):
        cv2.namedWindow("Select Contour")
        cv2.imshow("Select Contour",img)
        cv2.setMouseCallback("Select Contour",self.ret_point)
        matched_cnts = []
        matched_cnts_idx = []
        while(1):
            # Wait for User to Select Contour
            
            if self.pt!=None:
                for idx,cnt in enumerate(cnts):
                    ret = cv2.pointPolygonTest(cnt,self.pt,False)
                    if ret==1: # point is inside a contour ==> return its idx and cnt
                        matched_cnts.append(cnt)
                        matched_cnts_idx.append(idx)

                if matched_cnts==[]:
                    #logger.warning("(Incorrect Selection) --> Please only select an object!!")
                    self.pt = None # Reset point to None 
                elif len(matched_cnts) == 1:
                    cnt = matched_cnts[0]
                    idx = matched_cnts_idx[0]
                    if not Loop:
                        cv2.destroyWindow("Select Contour")
                    self.pt = None # Reset point to None                         
                    return idx,cnt
                else:
                    # find the biggest countour (c) by the area
                    cnt = min(matched_cnts, key = cv2.contourArea)
                    idx = matched_cnts.index(cnt)
                    if not Loop:
                        cv2.destroyWindow("Select Contour")
                    self.pt = None # Reset point to None                         
                    return idx,cnt

            k = cv2.waitKey(1)
            if k==27:
                #logger.warning("(Esc key pressed) --> No contours selected + Exiting!!!")
                return -1,-1 # idx = -1 (indicating no contour and exiting!)


def get_fileName(file_path):
    if not os.path.isfile(file_path):
        print(f"Not a file path...{file_path}\nCheck again!")
        return None

    file_basename = os.path.basename(file_path)
    fileName = os.path.splitext(file_basename)[0]
    return fileName
    
def get_data(topic, type = "img", folder_dir = None):

    if folder_dir==None:
        if topic =="tracking":
            folder_dir = "Data/NonFree/Advanced/Tracking/test_videos"
            fomrats = (".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm")
        elif topic =="multitracking":
            folder_dir = "Data/NonFree/Advanced/Tracking/multi_test_videos"
            fomrats = (".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm")


    data_dirs = []
    filenames = []
    for file in os.listdir(folder_dir):
        #if file.endswith(".jpg"):
        if file.endswith(fomrats):
            filename, _ = os.path.splitext(file)
            file_path = os.path.join(folder_dir, file)
            data_dirs.append(file_path)
            filenames.append(filename)
    return data_dirs,filenames


# ## Secure Access APP

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, 2, 1, 3}
        The (0, 1) position is at the top left corner,
        the (2, 3) position is at the bottom right corner
    bb2 : dict
        Keys: {0, 2, 1, 3}
        The (x, y) position is at the top left corner,
        the (2, 3) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def to_ltrd(rect,rect_format ="bbox"):
    if rect_format =="bbox":
        #  __ _ _ _ _ _  X axis
        # |     x1  y1
        # |   (left,top) ---------
        # |              |       |
        # |              |       |
        # |              --------- (right,down)
        #  Y axis                     x2   y2
        #           
        left, top, w, h = rect
        right = left + w
        down = top + h
        # Returns (x1, y1, x2  , y2)
        return ((left,top,right,down))
    elif rect_format =="css":
        (top, right, down, left) = rect
        return ((left,top,right,down))
 
def to_trbl(ltwh,scale=1):
    #  __ _ _ _ _ _  X axis
    # |     x1  y1
    # |   (left,top) ---------
    # |              |       |
    # |              |       |
    # |              --------- (right,down)
    #  Y axis                     x2   y2
    #           
    left, top, w, h = ltwh
    right = left + w
    bottom = top + h
    # Returns (x1, y1, x2  , y2)
    return ((int(top*scale),int(right*scale),int(bottom*scale),int(left*scale)))

def to_ltwh(rect,rect_format ="corners"):
    """Get current position in bounding box format `(top left x, top left y,
    width, height)`.

    Params
    ------
    orig : bool
        To use original detection (True) or KF predicted (False). Only works for original dets that are horizontal BBs.
    orig_strict: bool 
        Only relevant when orig is True. If orig_strict is True, it ONLY outputs original bbs and will not output kalman mean even if original bb is not available. 

    Returns
    -------
    ndarray
        The KF-predicted bounding box by default.
        If `orig` is True and track is matched to a detection this round, then the original det is returned.

    """
    if rect_format == "corners":
        left, top, right, bottom = rect
    elif rect_format == "css":
        top, right, bottom, left = rect # (top, right, bottom, left)
    else:
        print(f"\nUnknown format {rect_format}!")
        
    width = abs(right - left)  
    height = abs(bottom - top) 
    return ((left,top,width,height))        

# ## Secure Access APP


# def get_optimal_font_scale(text, width):
#     """ Returns the optical font Scale for text to fit in the rectangle 

#     Args:
#         text (String): _description_
#         width (int): _description_

#     Returns:
#         int: _description_
#     """
    
#     for scale in reversed(range(0, 120, 1)):
#         textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale/10, thickness=1)
#         new_width = textSize[0][0]
#         #print(new_width)
#         if (new_width <= int(width/2)):
#             return (scale/10),new_width
#     return 1,1

# def putText_bbox(img,text_list,orig_list,type = "bbox"):
    
#     fontScale,new_width = get_optimal_font_scale(text_list[0],img.shape[1])

#     clr_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
        
#     for idx,txt in enumerate(text_list):
#         org = orig_list[idx]
        
#         if type=="bbox":
#             clr = clr_list[idx]
#             if idx==0:
#                 # is even===> then adjust for pt center
#                 #print(f"adjusted .... YAyyy! at {idx}")
#                 org = (org[0]-int(new_width/2),org[1])
#             elif idx==3:
#                 # is even===> then adjust for pt center
#                 #print(f"adjusted .... YAyyy! at {idx}")
#                 org = (org[0]-int(new_width/2),org[1])
#         else:
#             clr = random.random(size=3) * 256

#         #cv2.putText(img,txt,org,cv2.FONT_HERSHEY_PLAIN,fontScale,clr,4 )
#         putText(img,txt,org,cv2.FONT_HERSHEY_PLAIN,1,clr,1 )



def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)



def draw_fancy_bbox(img, pt1, pt2, color, thickness, r=15, d=10,style = "fancy"):
    if style == "fancy":
        x1, y1 = pt1
        x2, y2 = pt2
        xr, yr = x1 + r, y1 + r
        xl, yl = x2 - r, y2 - r

        # Top left
        xrd, yrd = xr + d, yr + d
        cv2.line(img, (xr, y1), (xrd, y1), color, thickness)
        cv2.line(img, (x1, yr), (x1, yrd), color, thickness)
        cv2.ellipse(img, (xr, yr), (r, r), 180, 0, 90, color, thickness)

        # Top right
        xld, yrd = xl - d, yr + d
        cv2.line(img, (xl, y1), (xld, y1), color, thickness)
        cv2.line(img, (x2, yr), (x2, yrd), color, thickness)
        cv2.ellipse(img, (xl, yr), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        xrd, yld = xr + d, yl - d
        cv2.line(img, (xr, y2), (xrd, y2), color, thickness)
        cv2.line(img, (x1, yl), (x1, yld), color, thickness)
        cv2.ellipse(img, (xr, yl), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        xld, yld = xl - d, yl - d
        cv2.line(img, (xl, y2), (xld, y2), color, thickness)
        cv2.line(img, (x2, yl), (x2, yld), color, thickness)
        cv2.ellipse(img, (xl, yl), (r, r), 0, 0, 90, color, thickness)
        
    elif style == "dotted":
        pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
        drawpoly(img,pts,color,thickness,style)



image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
