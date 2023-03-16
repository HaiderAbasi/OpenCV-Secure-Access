import os
import cv2
import config
from recognition import FaceRecognizer
#Todo:
# a) Try to print the face extracted in create data embeddings to see if itwas done correctly


# Step the repo_dir based on where the file is run at, Expected to be in repo_dir/src/main.py
config.repo_dir = os.path.join(os.path.dirname(__file__), os.pardir)


# Step 1: Creating Face Recognizer object ....
face_recog = FaceRecognizer()

# Set the path to the directory containing the images
image_dir_path = r'data\test'

# Set the path to the image containing the individual to search for
search_image_path = r'data\authorized\haider\haider.jpg'

# Directory Listing the authorized personnel in their own folders 
#
# Authorized Personnel
# |
#  haider
#  |
#   1.jpg
#   2.jpg
# |
#  omer
#  |
#   1.jpg
#   2.jpg
#
dataset_dir = r"data\authorized"

# Get embeddings [encodings + Names]
data = face_recog.get_embeddings(dataset_dir)
print("data = ",data)
cv2.waitKey(0)


# Get the name of the person to find from the search image path
person_to_find = os.path.splitext(os.path.basename(search_image_path))[0]

# Create the directory to store the matched images
matched_dir_path = os.path.join(image_dir_path, f'matched_{person_to_find}')
if not os.path.exists(matched_dir_path):
    os.makedirs(matched_dir_path)

# Perform face verification on each image file in the directory and save the matched images
matching_images = []
img_1_objs = None
img_2_objs = None
for image_file in os.listdir(image_dir_path):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    image_path = os.path.join(image_dir_path, image_file)
    
    try:
        # Step # 1 & 2 (Detection + Alignment)
        #img_1_objs, img_2_objs = face_recog.get_faces(search_image_path,image_path)
        img_2_objs = face_recog.extract_faces(img=image_path, grayscale=False, enforce_detection=True, align=True)
    except:
        print("face not detectod")
    no_face_detected = 0
    # Step # 2 & 3 (Representation + Verfication)
    if img_1_objs is not None and img_2_objs is not None:
        result = face_recog.verify(img_1_objs, img_2_objs,normalization="base")
    elif data is not None and img_2_objs is not None:
        result = face_recog.verify(data, img_2_objs,normalization="base")
    else:
        no_face_detected+=1
        # Ignore the image if no face is detected
        print("No face was detected in the current image.\nSkipping....")
        continue

    if result['verified']:
        matching_images.append(image_path)
        filename = os.path.basename(image_path)
        new_path = os.path.join(matched_dir_path, filename)
        cv2.imwrite(new_path, cv2.imread(image_path))
        #cv2.imwrite(new_path, result['img_content'])
        print(f"Saved matched image {filename} to {matched_dir_path}")
    else:
        matching_images.append(image_path)
        filename = os.path.basename(image_path)
        new_path = os.path.join(r"data\test\not_matched", filename)
        cv2.imwrite(new_path, result['img_content'])
        print(f"{image_path} Face not matched with the Person of interest!")

print(f"Found {len(matching_images)} images containing the individual in {search_image_path}")

print(f"No face was detected in {no_face_detected} images")