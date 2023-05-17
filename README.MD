Secure Access System
--------------------

This is a Python-based secure access system that uses facial recognition and tracking to authenticate authorized personnel.


![alt text](https://github.com/HaiderAbasi/OpenCV-Secure-Access/blob/dev_haider_sadlib/assets/Secure_Acceess_demo.gif)


Features
------------
- Only allow device/premises access 🔓 to authorized individuals. 

- Annoy 😠 kids away by disrupting their mouse/keyboard commands. 
- Lockout crucial files/folders 🚫 from specified individuals. 
> (To-do)
    - Gather complete activity-data for selected users. 

Requirements
------------
To run this project, you need to have the following Python packages installed:
- For Ubuntu 
  - Install python 3.8 using the following steps [(Ref)](https://www.linuxcapable.com/install-python-3-8-on-ubuntu-linux/)
    ```console
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt install python3.8
    sudo apt install python3.8-distutils
    sudo apt install python3.8-dev
    sudo apt install python3.8-tk
    ```
  - Setup and activate virtual environment inside the repository
    ```console
    sudo apt install pip -y
    python3 -m pip install --user virtualenv
    python3 -m virtualenv sa_venv -p /usr/bin/python3.8
    source sa_venv/bin/activate
    ```
  - Install required modules using the following two commands.
    ```console
    pip install -r prequirements.txt
    pip install -r requirements.txt
    ```
  - Test the application by running core.py


- For Windows 

  - Download and install Python 3.8.10.
  - Setup a virtual environment inside the repository.
  - You can install these packages by running the following command:

    ```console
    pip install -r requirements.txt
    ```

Usage
-----
To activate the secure access system, call the `activate_sa` method of the `secure_access_cv` class with the following arguments:

- `vid_id`: The ID of the camera/webcam device to capture frames from or the path of the video file to use.
- `dataset_dir`: The path to the directory containing the folders of image/s of authorized personnel to use for training the face recognition model.

Here's an example:

```python
from core import secure_access_cv

sas = secure_access_cv()
sas.activate_sa(vid_id=0, dataset_dir="authorized_personnels")
```


File structure
------------
The main file of the secure access system is core.py, which contains the secure_access_cv class.

```diff
|
+──authorized_personnels > (Authorized personnel images)
│   ├──Person #1 > (folder storing image/s of person 1)
│   └──Person #2
| 
|──data > data folder containing test images and video for debugging
│  └──test
|
+──models  > (Detection + Recognition + Embeddings)
│  ├──dlib_face_recognition_resnet_model_v1.dat
│  ├──known-live_embeddings-face_enc
│  ├──mmod_human_face_detector.dat
│  ├──shape_predictor_5_face_landmarks.dat
│  └──shape_predictor_68_face_landmarks.dat
|
├──.gitignore
├──LICENSE
|
|──config.py > (control variables for secure access)
├──tree.MD
|
|──requirements.txt > (required modules to run)
├──core.spec > (File for pyinstaller to generate executable for app)
|
+──core.py    > (Main file to run Secure Access App)
├──multi_object_tracking.py
├──recognition.py
└──utilities.py
```

Standalone executable (pyinstaller) [Ref](https://github.com/ageitgey/face_recognition/issues/357)
---------------------------------------
```
# 1) Install pyinstaller module
pip install --upgrade pyinstaller

# 2) Configure core.spec according to your requirements

# 3) Create executable using pyinstaller
pyinstaller .\core.spec
```


Contribution
------------
If you'd like to contribute to this project, please fork the repository and make a pull request.

License
------------
This project is licensed under the MIT License - see the LICENSE file for details.


Thanks
------------
Thank you 👏 [ageitgey](https://github.com/ageitgey).

Your work has made using face recognition  in projects a walk in the park. 

Also, thanks to [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom)) for creating dlib and for providing the trained facial feature detection and face encoding models used in this library.

Keep the OpenSource torch lit! 🔥