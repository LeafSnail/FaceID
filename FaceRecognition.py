import cv2

def runFaceRecognitionApp():
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the camera
    cap = cv2.VideoCapture(1)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face and display the predicted label and confidence score
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Persoana', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check if the user pressed the Escape key
        if cv2.waitKey(1) == 27:
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

