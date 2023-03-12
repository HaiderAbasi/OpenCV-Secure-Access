import os
import cv2

from recognition import FaceRecognizer

# Step 1: Creating Face Recognizer object ....
face_recog = FaceRecognizer()

# Set the path to the directory containing the images
image_dir_path = r'data\test_imgs'

# Set the path to the image containing the individual to search for
search_image_path = r'data\poi\iron_man.jpg'

# Get the name of the person to find from the search image path
person_to_find = os.path.splitext(os.path.basename(search_image_path))[0]

# Create the directory to store the matched images
matched_dir_path = os.path.join(image_dir_path, f'matched_{person_to_find}')
if not os.path.exists(matched_dir_path):
    os.makedirs(matched_dir_path)

# Perform face verification on each image file in the directory and save the matched images
matching_images = []
for image_file in os.listdir(image_dir_path):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    image_path = os.path.join(image_dir_path, image_file)
    
    # Step # 1 & 2 (Detection + Alignment)
    img_1_objs, img_2_objs = face_recog.get_faces(search_image_path,image_path)
    
    if img_1_objs is not None and img_2_objs is not None:
        # If faces found:
        # Step # 2 & 3 (Representation + Verfication)
        result = face_recog.verify(img_1_objs, img_2_objs,normalization="base")
    else:
        # Ignore the image if no face is detected
        print("No face was detected in the current image.\nSkipping....")
        continue

    if result['verified']:
        matching_images.append(image_path)
        filename = os.path.basename(image_path)
        new_path = os.path.join(matched_dir_path, filename)
        cv2.imwrite(new_path, cv2.imread(image_path))
        print(f"Saved matched image {filename} to {matched_dir_path}")
    else:
        print("Face not matched with the Person of interest!")

print(f"Found {len(matching_images)} images containing the individual in {search_image_path}")