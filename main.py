import cv2

#To detect Hand
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Key,Controller

cap=cv2.VideoCapture(0)

#Need room for keyboard keys
cap.set(3,1680)

#720p HD Resolution set
cap.set(4,1080)

detector = HandDetector(detectionCon=0.8)
keys = [["1","2","3","4","5","6","7","8","9","0","-","+","BS"],
        ["Q","W","E","R","T","Y","U","I","O","P","[","]","|"],
        ["A","S","D","F","G","H","J","K","L",";","'"],
        ["Z","X","C","V","B","N","M",",",".","/"]]

finalText=""

Keyboard=Controller()

#def drawALL(img, buttonList):
#    for button in buttonList:
#        x, y = button.pos
#        w, h = button.size
#        #Creating a button(keys)(100,100)-origin
#        cv2.rectangle(img,button.pos,(x+w ,y+h),(255,0,255), cv2.FILLED)
#
#        #cv2.putText(photo,"text",(position-x,position-y),font,font-thickness,(rgb-color),Scale)
#        cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
#    return img

def drawALL(img, buttonList):
    imgNew=np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y=button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0],button.size[1]),l=30, t=5, rt=0,
         colorR=(255, 0, 255), colorC=(0, 255, 0))
        cv2.rectangle(imgNew, button.pos, (x+button.size[0],y+button.size[1]),(255,0,255),cv2.FILLED)
        cv2.putText(imgNew,button.text,(x+40,y+60),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)

    out= img.copy()
    alpha =0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out

class Button():
    def __init__(self,pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text 
        #Now each botton has these three attributes
        
        

buttonList=[]
#calling to create a button
for i in range(len(keys)):
    for j,key in enumerate(keys[i]):
        buttonList.append(Button([90*j+5,90*i+10], key))

while True:
    #Boiler cap for running Web Cap
    success, img= cap.read()

    #Finding hands and their coordinates(Landmark points)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    img=drawALL(img,buttonList)

    #Checking if there is hand or not
    if lmList:
        for button in buttonList:
            #We need to know the position of button and whether
            #our finger is there in that range or not
            x,y=button.pos
            w,h=button.size
            #our tip of the index fingr is landmark 8(Media pipe)
            #our tip of the ring finger is landmark 12(Media pipe)

            if x < lmList[8][0] < x+w and y<lmList[8][1]<y+h:
                #Changind button color(keys)(100,100)-origin
                cv2.rectangle(img,button.pos,(x+w ,y+h),(175,0,175), cv2.FILLED)

                #cv2.putText(photo,"text",(position-x,position-y),font,font-thickness,(rgb-color),Scale)
                cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                
                #detector finds lenght b/w ldmrk 8,12 and return three params
                #len,etc,etc to ignore other two we are using underscores
                l,_,_ = detector.findDistance(8,12,img,draw=False)
                
                #when clicked
                if l<40:
                    if(button.text=="BS"):
                        Keyboard.press(Key.backspace)
                        sleep(0.5)
                    else:
                        Keyboard.press(button.text)
                        cv2.rectangle(img,button.pos,(x+w ,y+h),(0,255,0), cv2.FILLED)
                        cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                        finalText+=button.text
                        sleep(0.5)
                    #print(button.text)
                    

    cv2.rectangle(img,(50,350),(700,450),(175,0,175), cv2.FILLED)
    cv2.putText(img,finalText,(60,450),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)



    cv2.imshow("Image", img)
    cv2.waitKey(1)

