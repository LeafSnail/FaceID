import cv2
import threading
from deepface import DeepFace

isAllowed = False

ref_img = cv2.imread("ref.jpg")

def Identify(frame):
    global isAllowed
    try:
        if DeepFace.verify(frame, ref_img.copy())['verified']:
            isAllowed = True
        else:
            isAllowed = False
    except ValueError:
        pass

def runFaceIdentificationApp():

    # Open the camera
    cap = cv2.VideoCapture(1)
    timer = 0
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if ret:
            if timer % 30 == 0:
                try:
                    threading.Thread(target=Identify, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            timer += 1

        if isAllowed:
            cv2.putText(frame, "Is Allowed", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Is NOT Allowed", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check if the user pressed the Escape key
        if cv2.waitKey(1) == 27:
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()