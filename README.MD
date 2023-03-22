Secure Access System
--------------------

This is a Python-based secure access system that uses facial recognition and tracking to authenticate authorized personnel.

Requirements
------------
To run this project, you need to have the following Python packages installed:

- opencv-contrib-python==4.4.0.46
- dlib
- face_recognition

You can install these packages by running the following command:

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

Contribution
------------
If you'd like to contribute to this project, please fork the repository and make a pull request.

License
------------
This project is licensed under the MIT License - see the LICENSE file for details.