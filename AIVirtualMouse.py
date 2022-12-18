import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wCam, hCam = 640, 480
#fps Variable
pTime=0
#frame Size reduction Variable
frameR = 100
#Fluiding variables
fluiding = 5
plocX, plocY = 0, 0
clocX ,clocY = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()



while True:
    # step-1 : Find hand landmarks
    success, img = cap.read()
    img=detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # step-2 : Find tip of index finger and middle finger
    #Index mode will move the finger
    if(len(lmList)!=0):
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        # step-3 : Check which fingers are up
        fingers = detector.fingersUp()
            #print(fingers)

        # step-4 : Only index Finger : Moving Mode
        cv2.rectangle(img,(frameR,frameR), (wCam-frameR , hCam-frameR), (255,0,255), 2)

        if fingers[1]==1 and fingers[2]==0:

            # Convert Coordinates to get the correct coordinates
            x3 = np.interp(x1, (frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR),(0,hScr))

            # step-5 : Making in Fluid(values)
            clocX = plocX + (x3- plocX) / fluiding
            clocY = plocY + (y3- plocY) / fluiding

            # step-6 : Move Mouse
            autopy.mouse.move(wScr-clocX,clocY)
            plocX, plocY = clocX, clocY
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)

        # step-7 : For Verifing the Mode of mouse
        if fingers[1]==1 and fingers[2]==1: 
            #if in clicking mode - find distace b/w fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            #print(length)
            if length < 35:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)

                #if distance is short - click mouse
                autopy.mouse.click()

    # step-8 : fix frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)

    # step-9 : display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
