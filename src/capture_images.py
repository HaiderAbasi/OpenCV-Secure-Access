import cv2
import time

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Delay in seconds between bursts
burst_delay = 5

# Number of images to capture in each burst
burst_size = 3

start = time.time()
counter = 0
# Loop to capture images
while True:
    counter+=1
    # Wait for the specified time before capturing the burst
    elapsed_time = int(time.time() - start) 
    time_crossed = elapsed_time % burst_delay ==0

    # Capture the burst
    for i in range(burst_size):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if time_crossed:
            print(f"The burst time {elapsed_time} has elapsed, Capturing images")
            # Save the image to disk
            cv2.imwrite("data\\test\\image_{}.jpg".format(counter), frame)

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
