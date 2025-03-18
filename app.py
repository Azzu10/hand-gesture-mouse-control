import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image (mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand detection
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            # Map hand coordinates to screen size
            screen_x = np.interp(x, (0, w), (0, screen_width))
            screen_y = np.interp(y, (0, h), (0, screen_height))

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Click detection: If thumb (landmark 4) is close to index finger
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            
            distance = np.hypot(x - thumb_x, y - thumb_y)
            
            if distance < 30:  # Adjust threshold for click detection
                pyautogui.click()
    
    # Show webcam output
    cv2.imshow("Hand Gesture Mouse", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
