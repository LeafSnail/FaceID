import random
import time
import cv2
import os
import threading
from deepface import DeepFace

isAllowed = False
takePicture = False
singleSampleFace = False
multiSampleFace = False
falsePositive = False
images = []

ref_img = cv2.imread("ref.jpg")

def load_images_from_folder():
    global images
    images = []
    path = "database/"
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)

def AddReference():
    print("Adding Reference")
    global takePicture
    takePicture = True
    load_images_from_folder()

def multiSampleFaceTest():
    global multiSampleFace
    multiSampleFace = True

def singleSampleFaceTest():
    global singleSampleFace
    singleSampleFace = True

def falsePositiveFaceTest():
    global falsePositive
    falsePositive = True

def Identify(frame):
    global isAllowed, images
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy())['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print(counter)

def SingleSampleTest(frame):
    global isAllowed, images

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("Facenet --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet512")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("Facenet512 --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "OpenFace")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("OpenFace --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "DeepFace")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("DeepFace --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "ArcFace")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("ArcFace --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "SFace")['verified']:
                isAllowed = True
                break
    except ValueError:
        pass
    print("SFace --- %s seconds ---" % (time.time() - start_time))

def MultiSampleTest(frame):
    global isAllowed, images

    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("Facenet: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet512")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("Facenet512: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "OpenFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("OpenFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "DeepFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("DeepFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "ArcFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("ArcFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "SFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("SFace: " + counter.__str__())
    print("--- %s seconds ---" % (time.time() - start_time))

def FalsePositiveTest(frame):
    global isAllowed, images

    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("Facenet: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "Facenet512")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("Facenet512: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "OpenFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("OpenFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "DeepFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("DeepFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "ArcFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("ArcFace: " + counter.__str__())

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    counter = 0
    try:
        isAllowed = False
        print("Identifying")
        for img in images:
            if DeepFace.verify(frame, img.copy(), "SFace")['verified']:
                counter = counter + 1
                isAllowed = True
    except ValueError:
        print("Failed: " + counter.__str__())
        pass
    print("SFace: " + counter.__str__())
    print("--- %s seconds ---" % (time.time() - start_time))

def runProjectApp():
    global takePicture, multiSampleFace, singleSampleFace, falsePositive
    load_images_from_folder()
    # Open the camera
    cap = cv2.VideoCapture(1)
    timer = 0
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if singleSampleFace:
            SingleSampleTest(frame)
            singleSampleFace = False

        if multiSampleFace:
            MultiSampleTest(frame)
            multiSampleFace = False

        if falsePositive:
            FalsePositiveTest(ref_img)
            falsePositive = False

        if takePicture:
            cv2.imwrite("database/" + random.randint(0, 9999999).__str__() + ".jpg", frame)
            takePicture = False

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