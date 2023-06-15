from tkinter import *
import threading
import Project

root = Tk()
root.title("Controller")

def onFalsePositiveButtonClick():
    Project.falsePositiveFaceTest()

def onMultiSampleTestButtonClick():
    Project.multiSampleFaceTest()

def onAddButtonClick():
    Project.AddReference()

def onSingleSampleTestButtonClick():
    Project.singleSampleFaceTest()

SSTestButton = Button(root, text="Single Sample Test", command=onSingleSampleTestButtonClick, bg="#00DDCA", height=1, width=20)
SSTestButton.grid(row=0, column=0)

MSTestButton = Button(root, text="Multi Sample Test", command=onMultiSampleTestButtonClick, bg="#00DDCA", height=1, width=20)
MSTestButton.grid(row=1, column=0)

AddButton = Button(root, text="Add Reference", command=onAddButtonClick, bg="#00DD33", height=1, width=20)
AddButton.grid(row=0, column=1)

FPTestButton = Button(root, text="False Positives Test", command=onFalsePositiveButtonClick, bg="#F00000", height=1, width=20)
FPTestButton.grid(row=1, column=1)

threading.Thread(target=Project.runProjectApp).start()

root.mainloop()
