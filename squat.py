import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV VideoCapture for the camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)

# Initialize variables
squats = 0
squat_state = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Pose
    results = pose.process(frame_rgb)

    # Check if any pose is detected
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        # Hip keypoints (11 and 12)
        hip1 = landmarks[11]
        hip2 = landmarks[12]

        # Calculate the vertical distance between the hip keypoints
        hip_distance = abs(hip1.y - hip2.y)

        # Check for a squat movement
        if hip_distance > 0.1 and not squat_state:
            squat_state = True
            squats += 1
            print("Squat Count:", squats)  # Print the count in the terminal
        elif hip_distance < 0.05:
            squat_state = False

        # Draw landmarks (optional)
        for landmark in landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display squat count on the camera screen
        cv2.putText(frame, "Squats: " + str(squats), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Squat Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
