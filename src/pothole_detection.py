#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import time
#Load YOLO
net = cv2.dnn.readNet("yolov4_training_final.weights","yolov4_testing.cfg") # Original yolov4
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
outputlayers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
 

 


# In[9]:


#loading image
thr = 0.4  # change this value according to your requirements
#input image path
frame = cv2.imread("5.jpg")

#cap=cv2.VideoCapture(video_name) #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
#starting_time= time.time()
#frame_id = 0
c = 0

detect = False
#ret,frame= cap.read() #
colors= np.random.uniform(0,255,size=(len(classes),3))
#if ret == False:
#    print(ret)
#    cap.release()    
#    cv2.destroyAllWindows()
#frame_id+=1
    
height,width,channels = frame.shape
#detecting objects
blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

        
net.setInput(blob)
outs = net.forward(outputlayers)
#print(outs[1])


#Showing info on screen/ get confidence score of algorithm in detecting an object in blob
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > thr:
            #object detected
            detect = True
            center_x= int(detection[0]*width)
            center_y= int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x=int(center_x - w/2)
            y=int(center_y - h/2)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            boxes.append([x,y,w,h]) #put all rectangle areas
            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
            class_ids.append(class_id) #name of the object tha was detected

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence= confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,4)
        cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

   # elapsed_time = time.time() - starting_time
   # fps=frame_id/elapsed_time
   # cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
if detect == True:
    cv2.imwrite("OUTPUT\crack"+ str(c)+".jpg", frame)
    c = c+1
        
    
cv2.imshow("Image",frame)
cv2.waitKey(0) #wait 1ms the loop will start again and we will process the next frame
# esc key stops the process
#     break;
#frame.release()    
cv2.destroyAllWindows()    


# In[11]:


# References: 
# https://www.youtube.com/watch?v=yGMZOD44GrI (Object Detection Using YOLO v4 on Custom Dataset | Practical Implementation- Code With Aarohi)


# In[12]:


# https://github.com/luisaugustos/Pothole-Recognition (Pothole Recognition using Drone Images - Luis Augusto Silva)


# In[ ]:


# https://github.com/AlexeyAB/darknet (Yolo v4 - AlexeyAB)

