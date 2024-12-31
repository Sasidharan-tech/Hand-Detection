import os
import cv2

# Define paths and parameters
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 4  # Number of classes you want to collect data for
dataset_size = 100  # Number of images to capture for each class

# Try opening the video capture
cap = cv2.VideoCapture(0)  # Change index if needed (0, 1, 2, etc.)
if not cap.isOpened():
    print("Error: Could not open video capture. Please check your camera.")
    exit()

# Loop through each class and collect data
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display a message to press 'Q' to start collecting images
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        cv2.putText(frame, 'Ready? Press "Q" to start collecting data!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Check if 'Q' key is pressed to start collecting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True

    # Collect dataset images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Show the frame while capturing
        cv2.imshow('frame', frame)
        # Save the image
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

        # Display how many images are left to collect
        print(f"Collecting image {counter}/{dataset_size} for class {j}")
        
        # Delay to allow time for the camera to update the frame
        cv2.waitKey(1)

    print(f"Finished collecting data for class {j}")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
