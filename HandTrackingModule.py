import cv2
import mediapipe as mp
import time
import math



class handDetector():
    ##################################################################################################################
    # def __int__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
    #     self.mode = mode
    #     self.maxHands = maxHands
    #     self.detectionCon = detectionCon
    #     self.trackingCon = trackingCon
    #
    #     self.mpHands = mp.solutions.hands
    #
    #     self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackingCon)
    #     self.mpDraw = mp.solutions.drawing_utils
    # ###############################################################################################################
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]






    def findHands(self, img, draw=True):
        #self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe only work with RGB so we need to
        # convert img(BGR // coz of open cv) to RGBimage

        self.results = self.hands.process(imgRGB)  # processing with mediapipe

        # print(results.multi_hand_landmarks)           # results.multi_hand_landmarks gives landmarks for the hand if no hand detected returns NONE

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:  # handlms is each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)  # drawing landmarks and connections
        return img

    def findPosition(self, img, handNo=0,draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList=[]                                                               # adding the coordinated in lmList
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]                  # now only printing for the given hand no
            for id, lm in enumerate(myHand.landmark):
                    # print(id,lm)                                                # we are getting id and co ordinate(in ratio of img height and width) of all the ids
                    h, w, c = img.shape                                             # gitting img sizr in pixels
                    cx, cy = int(lm.x * w), int(lm.y * h)                        # coverting ratio to pixels(direct coordinates)
                    xList.append(cx)
                    yList.append(cy)
                    # print(id, cx, cy)
                    self.lmList.append([id,cx,cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return (self.lmList,bbox)

    # def findPosition(self, img, handNo=0, draw=True):
    #     xList = []
    #     yList = []
    #     bbox = []
    #     self.lmList = []
    #     if self.results.multi_hand_landmarks:
    #         myHand = self.results.multi_hand_landmarks[handNo]
    #         for id, lm in enumerate(myHand.landmark):
    #             # print(id, lm)
    #             h, w, c = img.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             xList.append(cx)
    #             yList.append(cy)
    #             # print(id, cx, cy)
    #             self.lmList.append([id, cx, cy])
    #             if draw:
    #                 cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    #
    #         xmin, xmax = min(xList), max(xList)
    #         ymin, ymax = min(yList), max(yList)
    #         bbox = [xmin, ymin, xmax, ymax]
    #
    #         if draw:
    #             cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
    #                       (0, 255, 0), 2)
    #
    #     return self.lmList, bbox




    def fingersUp(self):
        fingers = []


        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  # for thumb check side ways.......only for right hand
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  # lower coordinates towards top
                fingers.append(1)
            else:

                fingers.append(0)

            # print(fingers)
        return fingers



    def findDistance(self, p1, p2, img, draw=True,r=7, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector=handDetector()         #obj
    while True:
        ret, img = cap.read()
        img= detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])


        # for FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # displaying text on output image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("webcam", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()