import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


video_capture = cv2.VideoCapture(0)

#All the students will be hardcoded in the code. Could be used as a seperate function
#but for initial trial it is hardcoded

musk_image = face_recognition.load_image_file("images\Elon Musk.jpg")
musk_encoding = face_recognition.face_encodings(musk_image)[0]

jeff_image = face_recognition.load_image_file("images\Jeff Bezoz.jpg")
jeff_encoding = face_recognition.face_encodings(jeff_image)[0]

messi_image = face_recognition.load_image_file("images\Messi.webp")
messi_encoding = face_recognition.face_encodings(messi_image)[0]

rahul_image = face_recognition.load_image_file("images\Rahul.jpg")
rahul_encoding = face_recognition.face_encodings(rahul_image)[0]

ravi_image = face_recognition.load_image_file("images\Ravi Shashtri.jpg")
ravi_encoding = face_recognition.face_encodings(ravi_image)[0]

rayan_image = face_recognition.load_image_file("images\Ryan Reynolds.jpg")
rayan_encoding = face_recognition.face_encodings(rayan_image)[0]

sachin_image = face_recognition.load_image_file("images\Sachin.jpg")
sachin_encoding = face_recognition.face_encodings(sachin_image)[0]

kohli_image = face_recognition.load_image_file("images\Virat Kohli.jpg")
kohli_encoding = face_recognition.face_encodings(kohli_image)[0]


known_face_encoding = [
musk_encoding,
jeff_encoding,
rayan_encoding,
sachin_encoding,
kohli_encoding,
ravi_encoding,
rahul_encoding,
messi_encoding
]

known_faces_names = [
"Elon Musk",
"Jeff Bezoz",
"Messi",
"Rahul Dravid",
"Ravi Shastri",
"Ryan Reynolds",
"Sachin",
"Virat Kohli"
]

students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[min(best_match_index, len(known_faces_names) - 1)]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()