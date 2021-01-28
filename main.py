import os
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt


#LandmarkIndex = [36, 40, 42, 45, 33, 60, 64]
#Landmark_to_Glasses_Coords = np.array([[417, 477], [516, 477], [615, 477], [715, 477], [576, 598], [544, 708], [632, 708]], np.float32)
LandmarkIndex = [36, 42, 64]
Landmark_to_Glasses_Coords = np.array([[417, 477], [615, 477], [632, 708]], np.float32)


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
        

def FaceLandmarkDetection(Image):
    # Initializing dlib's face detector (HOG-based) and then creating
    # the facial landmark predictor
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Converting image to grayscale
    Image_Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the image
    Rects = Detector(Image_Gray, 2)

    # Landmarks of all the faces will be stored in this variable
    All_Faces_Landmarks = []

    # Looping over the faces detected
    for (i, Rect) in enumerate(Rects):
        # Predicting facing landmarks
        Landmarks = Predictor(Image_Gray, Rect)

        # Extracting landmark coordinates from Landmarks detected
        Landmark_Coords = np.zeros((68, 2), dtype=np.float32)  # As 68 landmarks are detected
        for i in range(68):
            Landmark_Coords[i] = np.array([Landmarks.part(i).x, Landmarks.part(i).y], np.float32)

        # Storing the landmarks of the faces
        All_Faces_Landmarks.append(Landmark_Coords)

    return All_Faces_Landmarks


def ExtractRequiredLandmarks(FaceLandmarks):
    FinalFaceLandmarks = [Landmarks[LandmarkIndex] for Landmarks in FaceLandmarks]
    return FinalFaceLandmarks


def TransformImage(Image, Landmarks, a, b):
    # Finding homography matrix
    #(HomographyMatrix, Status) = cv2.findHomography(Landmark_to_Glasses_Coords, Landmarks, cv2.RANSAC, 4.0)
    HomographyMatrix = cv2.getAffineTransform(Landmark_to_Glasses_Coords, Landmarks)
    
    # Transforming the image
    TransformedImage = cv2.warpAffine(Image, HomographyMatrix, (b, a))

    return TransformedImage


def OverlapImages(BaseImage, SecImage):
    # Separating the mask
    Mask = SecImage[:, :, -1]
    Mask = cv2.cvtColor(Mask, cv2.COLOR_GRAY2BGR)
    SecImage = SecImage[:, :, :-1]

    # Overlaping the images
    BaseImage = cv2.bitwise_and(BaseImage, cv2.bitwise_not(Mask))
    OverlapedImage = cv2.bitwise_or(BaseImage, cv2.bitwise_and(SecImage, Mask))

    return OverlapedImage


if __name__ == "__main__":
    # Reading images with glasses to be overlaped and the face images
    GlassesImage = cv2.imread("PipeAndFace.png", cv2.IMREAD_UNCHANGED)
    Images, ImageNames = ReadImage("FaceImages")

    # Looping over face images and processing them
    for i in range(len(Images)):
        Image = Images[i]

        # Detecting face landmarks
        FaceLandmarks = FaceLandmarkDetection(Image)

        # Extracting the required landmarks for the problem.
        FaceLandmarks = ExtractRequiredLandmarks(FaceLandmarks)

        
        for Landmarks in FaceLandmarks:
            # Transforming glasses image for overlap
            TransformedGlassesImage = TransformImage(GlassesImage.copy(), Landmarks, Image.shape[0], Image.shape[1])

            Image = OverlapImages(Image, TransformedGlassesImage)

        plt.imshow(Image[:, :, ::-1])
        plt.show()
