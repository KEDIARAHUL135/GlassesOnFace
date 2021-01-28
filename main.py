import os
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt


def ReadImage(InputImagePath):
    Images = []                     # Input Images will be stored in this list.
    ImageNames = []                 # Names of input images will be stored in this list.
    
    # Checking if path is of file or folder.
    if os.path.isfile(InputImagePath):						    # If path is of file.
        InputImage = cv2.imread(InputImagePath)                 # Reading the image.
        
        # Checking if image is read.
        if InputImage is None:
            print("Image not read. Provide a correct path")
            exit()
        
        Images.append(InputImage)                               # Storing the image.
        ImageNames.append(os.path.basename(InputImagePath))     # Storing the image's name.

	# If path is of a folder contaning images.
    elif os.path.isdir(InputImagePath):
		# Getting all image's name present inside the folder.
        for ImageName in os.listdir(InputImagePath):
			# Reading images one by one.
            InputImage = cv2.imread(InputImagePath + "/" + ImageName)
			
            Images.append(InputImage)							# Storing images.
            ImageNames.append(ImageName)                        # Storing image's names.
        
    # If it is neither file nor folder(Invalid Path).
    else:
        print("\nEnter valid Image Path.\n")
        exit()

    return Images, ImageNames
        

def FaceDetection(Image):
    # Converting to grayscale
    Image_Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    # Reading cascade classifier file
    Face_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detecting faces
    Faces = Face_Cascade.detectMultiScale(Image_Gray, 1.3, 5)

    # Returning faces
    return Faces


if __name__ == "__main__":
    # Reading images with glasses to be overlaped and the face images
    GlassesImage = cv2.imread("PipeAndFace.png", cv2.IMREAD_UNCHANGED)
    Images, ImageNames = ReadImage("FaceImages")

    # Looping over face images and processing them
    for i in range(len(Images)):
        Image = Images[i]

        # Detecting faces in the image.
        Faces = FaceDetection(Image)

        for face in Faces:
            (x, y, w, h) = face
            Image = cv2.rectangle(Image, (x, y), (x+w, y+h), (0, 255, 0), 3)

        plt.imshow(Image[:, :, ::-1])
        plt.show()