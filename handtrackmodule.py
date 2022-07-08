import math
import time
import cv2
import mediapipe as mp

    
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        xList = []
        yList = []
        bbox = []
        ###################################################################################################
        # THIS CODE IS FOR TRACKING JUST ONE HAND
        ###################################################################################################

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])

                # If you wanna draw circles on each point then uncomment this.
                # if draw:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0), 2)
        ###################################################################################################


        ###################################################################################################
        # THIS CODE IS FOR TRACKING MORE THAN ONE HAND (I.E. TWO HANDS AS THE MAX-NUMBER OF HANDS IS TWO).
        ###################################################################################################
        # if self.results.multi_hand_landmarks:

        # myHand = []
        # for i in range(0, len(self.results.multi_hand_landmarks)):
        #     myHand.append(self.results.multi_hand_landmarks[i])

        #     for id, lm in enumerate(myHand[i].landmark):
        #         h, w, c = img.shape
        #         cx, cy = int(lm.x * w), int(lm.y * h)

        #         self.lmList.append([i, id, cx, cy])

        #         if draw:
        #             cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        ###################################################################################################

        return self.lmList

    def fingersUp(self):
        tipIds = [4, 8, 12, 16, 20]

        fingers = []

        if len(self.lmList) != 0:

            # This will work only for left hand, if you wanna work with both the hands then you gotta detect if the hand is right or left before and then if it is left, change the comparision sign from thumb's condition

            # Thumb
            if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        fingers = detector.fingersUp()
        print(fingers)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_ITALIC, 1, (235, 106, 0), 3)

        cv2.imshow("Image", img)

        if (cv2.waitKey(1) == 27) or (not (cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE))):
            break


if __name__ == '__main__':
    main()
