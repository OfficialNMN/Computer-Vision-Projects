import cv2

cap=cv2.VideoCapture(0)

# setting width, height, brightness
cap.set(3,1280)
cap.set(4,720)
cap.set(10,150)

# creating a list of classnames from coco.names
classnames=[]
classfile='coco.names'
with open(classfile,'rt') as f:
    classnames=f.read().rstrip('\n').split('\n')

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

# initialising a pre trained deep neural network with given weight file and configuration file
net= cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    _,img=cap.read()
    # return the classID, confidence and BoundingBox
    classIds,confs,bbox=net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    # this will only run when something is detected
    if len(classIds)!=0:
        # looping in 3 lists simultaneously using zip
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            # creating a rectangle around detected item
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            # writing the name of the item detected
            cv2.putText(img,classnames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # writing the confidence percent with which the object is detected
            cv2.putText(img,str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Ouput",img)
    cv2.waitKey(1)



