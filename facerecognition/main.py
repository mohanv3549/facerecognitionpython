import face_recognition
import csv
import cv2
import numpy as np
from datetime import datetime

videocapture = cv2.VideoCapture(0)

virat = face_recognition.load_image_file('knownfaces/virat.png')
virat_encode = face_recognition.face_encodings(virat)[0]
ab = face_recognition.load_image_file('knownfaces/ab.png')
ab_encode = face_recognition.face_encodings(ab)[0]
max = face_recognition.load_image_file('knownfaces/max.png')
max_encode = face_recognition.face_encodings(max)[0]

known_face_encodings=[virat_encode,ab_encode,max_encode]
known_face_names=["virat","ab","maxwell"]

students=known_face_names.copy()
face_locations=[]
face_encodings=[]

now=datetime.now()
currentdate=now.strftime("%y-%m-%d")

f=open(f"{currentdate}.csv","w")
lnwriter=csv.writer(f)

while True:
    _,frames=videocapture.read()
    smallFrame=cv2.resize(frames,(0,0,),fx=0.25,fy=0.25)
    rgb_smallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_RGB2BGR)

    face_locations=face_recognition.face_locations(rgb_smallFrame)
    face_encodings=face_recognition.face_encodings(rgb_smallFrame,face_locations)

    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index=np.argmin(face_distance)
        name=""
        if(best_match_index):
            name=known_face_names[best_match_index]
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomlefttext=(10,100)
            fontScale=1.5
            fontColor = (255,0,0)
            thickness=3
            lineType=2
            cv2.putText(frames,name + "present",bottomlefttext,font,fontScale,fontColor,thickness,lineType)

        if name in students:
            students.remove(name)
            current_time=now.strftime("%H-%M-%S")
            lnwriter.writerow([name,current_time])


    cv2.imshow("attendance",frames)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
videocapture.release()
cv2.destroyAllWindows()
f.close()
