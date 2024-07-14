import cv2
import mediapipe
import time

video = cv2.VideoCapture(0)

# Initalize hand landmarks
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands()
mp_draw = mediapipe.solutions.drawing_utils

pTime = 0
cTime = 0
tip_landmarks = [4, 8, 12, 16, 20]

# Start a Video 
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        print("problem")
        break

    # Flip 180 degree
    frame = cv2.flip(frame, 1)

    # Convert scale to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image to get landmarks
    results = hands.process(rgb_frame)

    # Extract landmarks
    hand_landmarks = results.multi_hand_landmarks

    # Iterate over landmarks
    if hand_landmarks != None:
        for hand_lms in hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)


    # Calculate frames per second
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(f"{int(fps)} fps"), (40, 40), 1, cv2.FONT_HERSHEY_COMPLEX, (0,0,255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

