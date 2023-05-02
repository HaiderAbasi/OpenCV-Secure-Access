"""
Functions related to security and access control.
"""
import pyautogui
import keyboard
import random
import cv2
import ctypes
import os
import atexit
import subprocess
import time
import platform

import concurrent.futures
from multiprocessing import Value

from utilities import draw_fancy_bbox


class SecurityActions:
    def __init__(self):
        self.mode = ""
        
        self.lock_iter = 0
        self.max_wait = 50

        self.default_keyboard_layout = None
        self.exec_annoy = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.futures_annoy = None
        self.enable_annoy = Value('i', 0)
        
        self.folder_hidden = False
        self.folder_path = r"D:\Haider\Udemy\Private"
        
        
        # Register a function to unhide the folder when the program exits
        atexit.register(self.lockout_access,False)
        atexit.register(self.__stop_annoying)

        
    def perform_security_actions(self,image):
        if self.mode == "lock":
            self.logout(image)
        elif self.mode == "annoy":
            if not self.futures_annoy or not self.futures_annoy.running():
                self.enable_annoy.value = 1 # Annoy the kids!
                self.futures_annoy = self.exec_annoy.submit(self.move_mouse_randomly)
        elif self.mode == "restrict":
            if not self.folder_hidden:
                self.lockout_access()
            self.folder_hidden = True

        
        if self.mode!= "annoy":
            if self.futures_annoy and self.futures_annoy.running():
                self.enable_annoy.value = 0 # Disable annoyance
        

        if self.mode!= "restrict" and self.folder_hidden:
            self.lockout_access(False)
            self.folder_hidden = False
                
                
        
    def logout(self,image):
        s=(5,5)
        e=(image.shape[1]-5,image.shape[0]-5)
        draw_fancy_bbox(image,s,e,(0,0,255),4,style ='dotted')
        cv2.putText(image,f">>> Unauthorized access <<<",(int(image.shape[1]/2)-155,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(128,0,128),2)
        cv2.putText(image, f"Locking in {self.max_wait-self.lock_iter} s" ,(20,60) ,cv2.FONT_HERSHEY_DUPLEX, 1, (128,0,255))
        self.lock_iter = self.lock_iter + 1
        
        r,c = image.shape[0:2]
        r_s = int(r*0.2)
        r_e = r - r_s
        
        t_elpsd_p = 1 - ((self.max_wait - self.lock_iter)/self.max_wait)
        r_ln = r_e - r_s
        
        cv2.rectangle(image,(50,r_s),(80,r_e),(0,0,128),2)
        
        cv2.rectangle(image,(52,r_s + int(r_ln*t_elpsd_p)),(78,r_e),(0,140,255),-1)
        
        if self.lock_iter== self.max_wait:
            self.lock_iter = 0
            ctypes.windll.user32.LockWorkStation()


    def __stop_annoying(self):
        if self.futures_annoy and self.futures_annoy.running():
            self.enable_annoy.value = 0 # Disable annoyance

    def __set_keyboard_layout(self,layout):
        self.default_keyboard_layout = '0' + hex(ctypes.windll.user32.GetKeyboardLayout(0))[2:]
        #print("self.default_keyboard_layout ",self.default_keyboard_layout)
        
        # Map the string layout to a hex value for LoadKeyboardLayoutW
        layout_map = {
            'us': '04090409',
            'uk': '00000809',
            'fr': '0000040c'}
        hex_layout = layout_map.get(layout)
        if hex_layout is None:
            raise ValueError(f"Unsupported keyboard layout: {layout}")
        
        if platform.system() == "Windows":
            ctypes.windll.user32.LoadKeyboardLayoutW(hex_layout, 1)
        elif platform.system() == "Linux":
            subprocess.call(['setxkbmap', hex_layout])
        else:
            print("Unsupported platform")
            
    def __reset_keyboard_layout(self):        
        if platform.system() == "Windows":
            #print("self.default_keyboard_layout ",self.default_keyboard_layout)
            current_layout = ctypes.windll.user32.GetKeyboardLayout(0)
            ctypes.windll.user32.UnloadKeyboardLayout(current_layout)
            ctypes.windll.user32.LoadKeyboardLayoutW(self.default_keyboard_layout, 1)   
            #print(f"Resetted keyboard to {self.default_keyboard_layout}")     
        elif platform.system() == "Linux":
            subprocess.call(['setxkbmap', '-layout', 'us'])
        else:
            print("Unsupported platform")
            
    def move_mouse_randomly(self):
        pyautogui.PAUSE = 0.0  # Set PyAutoGUI's pause time to 0 to speed up the loop
        self.__set_keyboard_layout('fr')  # Set the keyboard layout to US
        try:
            while bool(self.enable_annoy.value):
                # Ignore any keyboard or mouse events that occur while waiting for Esc
                while keyboard.is_pressed('esc'):
                    pass
                pyautogui.FAILSAFE = False
                pyautogui.PAUSE = 0.0
                if bool(self.enable_annoy.value):
                    x_offset = random.randint(-200, 200)  # Generate a random x offset between -10 and 10
                    y_offset = random.randint(-200, 200)  # Generate a random y offset between -10 and 10
                    pyautogui.moveRel(x_offset, y_offset)  # Move the mouse with the random offsets
                    time.sleep(3)
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.1
        except Exception as e:
            print(f"An error occurred: {e}")
        # Ignore any keyboard or mouse events that occurred while waiting for Esc
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self.__reset_keyboard_layout()  # Reset the keyboard layout to the original
        #print("Finished ....")
        

    def __lock_and_hide_folder(self,lock=True):
        try:
            if lock:
                subprocess.call(['attrib', '+h', '+s', self.folder_path])
                subprocess.call(['cacls', self.folder_path, '/e', '/p', '%s:n' % os.getlogin()])
            else:
                subprocess.call(['cacls', self.folder_path, '/e', '/p', '%s:f' % os.getlogin()])
                subprocess.call(['attrib', '-h', '-s', self.folder_path])
        except Exception as e:
            print(f"An error occurred while hiding/unhiding {self.folder_path}: {e}")

    def lockout_access(self, hide=True):
        try:
            if hide:
                self.__lock_and_hide_folder(True)  # 0x02 is the code for hidden attribute
            else:
                self.__lock_and_hide_folder(False)
        except Exception as e:
            print(f"An error occurred while hiding/unhiding {self.folder_path}: {e}")
            
        
        
    def __getstate__(self):
        d = self.__dict__.copy()
        # Delete all unpicklable attributes.
        del d['exec_annoy']
        del d['futures_annoy']
        del d['enable_annoy']
        return d


