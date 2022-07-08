import cv2
import time
import numpy as np
import handtrackmodule as ht
import autopy

# ##################################################################### #
# NOW THE MAIN QUESTION IS : HOW TO RUN IT WITHOUT SHOWING THE CAMERA!!
# ##################################################################### #

wCam, hCam = 1280, 1080

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = ht.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

frameR = 130  # FRAME REDUCTION

smoothening = 7

# plocX = PREVIOUS LOCATION OF X; plocY = PREVIOUS LOCATION OF Y;
plocX, plocY = 0, 0

# clocX = CURRENT LOCATION OF X; clocY = CURRENT LOCATION OF Y;
clocX, clocY = 0, 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 1. FIND THE HAND LANDMARKS
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)

    # 2. GET THE TIP OF THE INDEX AND MIDDLE FINGER
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        print(x1, y1, x2, y2)

        # 3. CHECK WHICH FINGERS ARE UP
        fingers = detector.fingersUp()
        print(fingers)

        # THIS IS THE HAND MOVING RANGE
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR * 4), (255, 0, 255), 2)

        try:
            # 4. ONLY INDEX FINGER : MOVING MODE
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert coordinates
                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam-frameR*4), (0, hScr))

                # 6. SMOOTHEN VALUES (Didn't understand this part)
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. MOVE MOUSE
                autopy.mouse.move(x3, y3)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. BOTH INDEX AND MIDDLE FINGERS ARE UP: CLICKING MODE
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                print(length)

                # 9. FIND DISTANCE BETWEEN THESE FINGERS (IF THE DISTANCE IS SHORT, THEN WE ARE GOING TO CLICK)
                if length < 47:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

                    # 10. CLICK MOUSE IF DISTANCE IS SHORT.
                    autopy.mouse.click()

        # Haven't figured out what to add or maybe there is no need to
        except:
            pass

    # 11. FRAME RATE
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. DISPLAY
    cv2.imshow("Image", img)

    if (cv2.waitKey(1) == 27) or (not (cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE))):
        break

