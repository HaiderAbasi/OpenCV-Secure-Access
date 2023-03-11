import cv2

# Load the image
img = cv2.imread(r'data\poi\thor.jpg')

# Display the image
cv2.imshow('Thor', img)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()