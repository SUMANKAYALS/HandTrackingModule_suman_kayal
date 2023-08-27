import cv2
import mediapipe as mp
import pyautogui

screen_width, screen_height = pyautogui.size()
index_y = 0
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgbFrame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            landmarks = handLms.landmark
            for id, landmarks in enumerate(landmarks):
                x = int(landmarks.x*frame_width)
                y = int(landmarks.y*frame_height)
                print(x, y)
                if id == 8:
                    cv2.circle(img=frame, center=(x,y), radius=20, color=(0, 255, 0))
                    index_x = screen_width / frame_width*x
                    index_y = screen_height / frame_height*y
                    pyautogui.moveTo(index_x, index_y)
                if id == 4:
                    cv2.circle(img=frame, center=(x,y), radius=20, color=(0, 255, 0))
                    thumb_x = screen_width / frame_width*x
                    thumb_y = screen_height / frame_height*y
                    # pyautogui.moveTo(index_x, index_y)
                    print(abs(index_y - thumb_y) )
                    if abs(index_y - thumb_y) < 20:
                        # print("click")
                        pyautogui.click()
                        pyautogui.sleep(1)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

