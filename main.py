import os
import cv2
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
        


if __name__ == "__main__":

    GlassesImage = cv2.imread("PipeAndFace.png", cv2.IMREAD_UNCHANGED)
    Images, ImageNames = ReadImage("FaceImages")

    for i in range(len(Images)):
        Image = Images[i]

        # Continue with code here