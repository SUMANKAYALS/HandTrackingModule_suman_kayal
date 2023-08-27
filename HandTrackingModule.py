import cv2
import mediapipe as mp
import time
import numpy as np




class HandDetector():
        # def __init__(self,mode=False,maxHands=2,detectionCon=0.5,modelComplexity=1,trackcon=0.5):
        def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackcon=0.5):
              self.mode=mode
              self.maxHands=maxHands
              self.detectionCon = detectionCon
              # self.modelComplexity =modelComplexity
              self.trackcon = trackcon
              self.mpHands = mp.solutions.hands
              self.hands = self.mpHands.Hands(static_image_mode =self.mode,max_num_hands = self.maxHands,min_detection_confidence = self.detectionCon,min_tracking_confidence = self.trackcon)
              # self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity,self.detectionCon,self.trackcon)
              self.mpDraw = mp.solutions.drawing_utils
              self.tipIds = [4,8,12,16,20]
              self.fingers = [] 
              self.lmlist = []

        def findHand(self,frame,draw=True,flipType=True):
               imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               self.results = self.hands.process(imgRGB)
               allHands = []
               h,w,c = frame.shape
               if self.results.multi_hand_landmarks:
                      for handType,handLms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
                             myHand = {}
                             mylmlist = []
                             xlist = []
                             ylist = []
                             for id, lm in enumerate(handLms.landmark):
                                    px,py,pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                                    mylmlist.append([px,py,pz])
                                    xlist.append(px)
                                    ylist.append(py)
                            
                             xmin, xmax = min(xlist), max(xlist)
                             ymin, ymax = min(ylist), max(ylist)
                             boxW, boxH = xmax - xmin, ymax - ymin
                             bbox = xmin, ymin, boxW, boxH
                             cx, cy = bbox[0] + (bbox[2] // 2), \
                                      bbox[1] + (bbox[3] // 2)

                             myHand["lmlist"] = mylmlist
                             myHand["bbox"] = bbox
                             myHand["center"] = (cx, cy)

                             if flipType:
                                    if handType.classification[0].label == "Right":
                                           myHand["type"] = "Left"
                                    else:
                                           myHand["type"] = "Right"
                             else:
                                    myHand["type"] = handType.classification[0].label
                             allHands.append(myHand)
                                    
                             if draw:
                                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                                    cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),
                                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),(255,0,255),2)
                                    cv2.putText(frame, myHand["type"],(bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
              
               if draw:
                      return allHands, frame
               else:
                      return allHands
       
        def fingersUp(self,myHand):
              myHandType = myHand["type"]
              mylmlist = myHand["lmlist"]
              if self.results.multi_hand_landmarks:
                     fingers = []

                     if myHandType == "Right":
                            if mylmlist[self.tipIds[0]][0] > mylmlist[self.tipIds[0] - 1][0]:
                                   fingers.append(1)
                            else:
                                   fingers.append(0)
                     else:
                            if mylmlist[self.tipIds[0]][0] < mylmlist[self.tipIds[0] - 1][0]:
                                   fingers.append(1)
                            else:
                                   fingers.append(0)
                     
                     for id in range(1,5):
                            if mylmlist[self.tipIds[id]][1] < mylmlist[self.tipIds[id] - 2][1]:
                                   fingers.append(1)
                            else:
                                   fingers.append(0)
              return fingers
       
        def findDistance(self,p1,p2,frame=None):

              x1, y1 = p1
              x2, y2 = p2
              cx, cy = (x1 + x2) // 2, (y1 +y2) // 2
              length = math.hypot(x2 - x1, y2 - y1)
              info = (x1,y1,x2,y2,cx,cy)
              if frame is not None:
                     cv2.circle(frame,(x1,y1),15,(255,0,255),cv2.FILLED)
                     cv2.circle(frame,(x2,y2),15,(255,0,255),cv2.FILLED)
                     cv2.line(frame,(x1,y1),(x2,y2),15,(255,0,255),3)
                     cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED)
                     return length,info,frame
              else:
                     return length,info
                      
                
                 

       # def findPosition(self,frame,handNo=0, draw= True):
       #         lmlist = []
       #         if self.results.multi_hand_landmarks:
       #               myHend=self.results.multi_hand_landmarks[handNo]
       #               for id, lm in enumerate(myHend.landmark):
       #                      h , w, c = frame.shape
       #                      cx, cy= int(lm.x * w), int(lm.y * h)
       #                      # print(id,cx,cy)
       #                      lmlist.append([id,cx,cy])
       #                      if draw:
       #                             cv2.circle(frame,(cx,cy),15,(255,0,255), cv2.FILLED)
       #         return lmlist        
               
# cap = cv2.VideoCapture(0)



# pTime = 0
# cTime = 0
# while True:
#         ret, frame = cap.read()







                
        # cv2.imshow("frame",frame)
        # if cv2.waitKey(1) == ord('q'):
        #         break

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,12,255),3)
# # # cap.release()
# # cv2.destroyAllWindows()
#         cv2.imshow("frame",frame)
#         cv2.waitKey(1)


def main():
       pTime = 0
       cTime = 0
       cap = cv2.VideoCapture(0)
       detector = HandDetector(detectionCon=0.8, maxHands=2)
       while True:
              
              # success, img = cap.read()
              ret, frame = cap.read()
              hands, frame = detector.findHand(frame)
              # lmlist = detector.findPosition(frame)
              # if len(lmlist)!=0:
              #        print(lmlist[4])
              if hands:
                     hand1 = hands[0]
                     lmlist1 = hand1["lmlist"]
                     bbox1 = hand1["bbox"]
                     centerpoint1 = hand1["center"]
                     handType1 = hand1["type"]

                     fingers1 = detector.fingersUp(hand1)
                     if len(hands) == 2:
                            hand2 = hand1
                            lmlist2 = hand2["lmlist"]
                            bbox2 = hand2["bbox"]
                            centerpoint2 = hand2["center"]
                            handType2 = hand2["type"]
                            fingers2 = detector.fingersUp(hand2)

                            
                            # length, info, frame = detector.findDistance(lmlist1[0][0:2], lmlist2[8][0:2], frame)


              # cTime = time.time()
              # fps = 1 / (cTime - pTime)
              # pTime = cTime

              # cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
# # cap.release()
# cv2.destroyAllWindows()
              cv2.imshow("frame",frame)
              cv2.waitKey(1)    




if __name__ == "__main__":

    main()