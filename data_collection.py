# Import necessary libraries
import mediapipe as mp
import numpy as np
import cv2

# Function to check if the body is in frame
def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
        return True  # Return True if body is in frame
    return False  # Return False if body is not in frame

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Get the name of the asana from the user
name = input("Enter the Name of the Asana : ")

# Initialize the Pose object from the Mediapipe library
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# Initialize an empty list to store the landmarks
X = []
data_size = 0

# Start the main loop
while True:
    lst = []
    _, frm = cap.read()  # Read a frame from the video capture
    frm = cv2.flip(frm, 1)  # Flip the frame horizontally
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  # Process the frame with the Pose object

    # Check if the body is in frame and if landmarks are available
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)  # Append normalized x-coordinate
            lst.append(i.y - res.pose_landmarks.landmark[0].y)  # Append normalized y-coordinate
        X.append(lst)  # Append the list of landmarks to the main list
        data_size = data_size + 1  # Increment the data size counter

    # If body is not in frame or landmarks are not available
    else:
        cv2.putText(frm, "Make Sure Full body visible", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)  # Display a warning message

    # Draw the landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)  # Display the data size counter
    cv2.imshow("window", frm)  # Display the frame

    # Break the loop if the 'Esc' key is pressed or the data size exceeds 80
    if cv2.waitKey(1) == 27 or data_size > 80:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save the landmarks data to a NumPy file
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)  # Print the shape of the landmarks data
