import cv2
import face_recognition

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image of known face along with the corresponding name
known_face_image_path = "images/WIN_20240206_14_00_34_Pro.jpg"
known_face_name = "Ashen"

# Compute face encoding for the known face
known_face_image = face_recognition.load_image_file(known_face_image_path)
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

# Load webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2280)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1720)  # Set height to 720 pixels

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)[0]

        name = known_face_name if match else "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)

    # Check if the 'q' key is pressed to close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
