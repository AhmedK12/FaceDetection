import cv2
face_classifier = cv2.CascadeClassifier('/root/Pictures/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')


def faces_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return None
    for(x, y, h, w) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if faces_extractor(frame) is not None:
        count += 1
        face = cv2.resize(faces_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = '/root/Desktop/face/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print('Face Not Found')
        pass
    if cv2.waitKey(1) == 13 or count == 900:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Sample Complete!!!')
