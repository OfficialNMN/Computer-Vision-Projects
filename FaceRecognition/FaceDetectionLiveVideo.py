import cv2 as cv

capture=cv.VideoCapture(0)
haar_cascade=cv.CascadeClassifier('haar_face.xml')

while True:
    isTrue,frame=capture.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=3)
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv.imshow('DetectedFaces', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()