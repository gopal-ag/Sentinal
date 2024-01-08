import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  

exit_direction = None
shoulder_diff_history = []  
hip_y_history = []  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate shoulder difference and smooth using history
        shoulder_diff = left_shoulder.x - right_shoulder.x
        shoulder_diff_history.append(shoulder_diff)
        if len(shoulder_diff_history) > 10:  
            shoulder_diff_history.pop(0)
        smoothed_shoulder_diff = sum(shoulder_diff_history) / len(shoulder_diff_history)


        hip_y_history.append(left_hip.y)
        if len(hip_y_history) > 20:  
            hip_y_history.pop(0)
        hip_movement_trend = (hip_y_history[-1] - hip_y_history[0]) / len(hip_y_history)


        head_direction = (
            "Looking Left" if nose.x < (left_shoulder.x + right_shoulder.x) / 2 else "Looking Right"
        )


        if head_direction == "Looking Left":
            orientation = "Facing Left" if smoothed_shoulder_diff > 0 else "Front"
        elif head_direction == "Looking Right":
            orientation = "Facing Right" if smoothed_shoulder_diff < 0 else "Front"
        else:
            orientation = (
                "Facing Left" if smoothed_shoulder_diff > 0 else "Facing Right"
            )

        exit_direction = (
            "b" if left_hip.y < right_hip.y and orientation == "Front" else "f"
        )
        if orientation != "Front":
            exit_direction = 'l' if left_hip.x < right_hip.x else 'r'


        cv2.putText(frame, f"Orientation: {orientation}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # cv2.putText(frame, f"Exit Direction: {exit_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
