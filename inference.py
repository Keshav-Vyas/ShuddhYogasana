# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Function to check if the body is in frame
def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
        return True  # Return True if body is in frame
    return False  # Return False if body is not in frame

# Load the trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe Pose object
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    lst = []
    _, frm = cap.read()  # Read a frame from the video capture
    window = np.zeros((940,940,3), dtype="uint8")  # Create a window for displaying the output
    frm = cv2.flip(frm, 1)  # Flip the frame horizontally
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  # Process the frame with the Pose object
    frm = cv2.blur(frm, (4,4))  # Apply a blur effect to the frame

    # Check if the body is in frame and if landmarks are available
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)  # Append normalized x-coordinate
            lst.append(i.y - res.pose_landmarks.landmark[0].y)  # Append normalized y-coordinate
        lst = np.array(lst).reshape(1,-1)  # Reshape the landmark data
        p = model.predict(lst)  # Make a prediction using the trained model
        pred = label[np.argmax(p)]  # Get the predicted label

        # Display the predicted label on the window
        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(window, pred , (180,180), cv2.FONT_ITALIC, 1.3, (0,255,0),2)
        else:
            cv2.putText(window, "Asana is either wrong not trained" , (100,180), cv2.FONT_ITALIC, 1.8, (0,0,255),3)

    # If body is not in frame or landmarks are not available
    else:
        cv2.putText(frm, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)  # Display a warning message

    # Draw the landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))

    # Display the frame in the window
    window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))
    cv2.imshow("window", window)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
