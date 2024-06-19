import cv2
import random
from imutils import face_utils

ids = {
    
}

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    if img is None:
        return None
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id, conf = clf.predict(gray_img[y:y+h, x:x+w])
        if conf <= 50:
            if id in ids:
                cv2.putText(img, ids[id], (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                return ids[id]
            else:
                cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return "Unknown"

def recognize(img, clf, faceCascade):
    if img is None:
        return None
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    name = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return name, img

def blink_eyes(img, eyes, coords):
    if len(eyes) >= 2:
        cv2.putText(img, 'Blink 5 times', (coords[0], coords[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def verify_smile(img, smile, coords):
    if len(smile) > 0:
        cv2.putText(img, 'Smile', (coords[0], coords[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)

blink_count = 0
eyes_detected = True
tasks_completed = False
random_blink_count = random.randint(3, 7)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from the video stream.")
        break

    name, frame = recognize(frame, clf, faceCascade)
    if name != "Unknown":
        if not tasks_completed:
            if eyes_detected:
                cv2.putText(frame, f'Get closer and blink {random_blink_count} times and smile', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                for (x, y, w, h) in faces:
                    if w > 250 :
                        roi_gray = gray[y:y+h, x:x+w]
                        eyes = eyeCascade.detectMultiScale(roi_gray)
                        smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.16, minNeighbors=35, minSize=(25, 25), flags=cv2.CASCADE_SCALE_IMAGE)
                        verify_smile(frame, smile, (x, y, w, h))
                        if len(eyes) == 0:
                            eyes_detected = False
                        else:
                            eyes_detected = True
                        if blink_count >= random_blink_count and len(smile) > 0:
                            tasks_completed = True
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                for (x, y, w, h) in faces:
                    if w > 250 :
                        roi_gray = gray[y:y+h, x:x+w]
                        eyes = eyeCascade.detectMultiScale(roi_gray)
                        if len(eyes) > 0:
                            blink_count += 1
                            eyes_detected = True
                        else:
                            eyes_detected = False
        else:
            cv2.putText(frame, 'Liveliness Verified', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f'Blink Count: {blink_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("face detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
