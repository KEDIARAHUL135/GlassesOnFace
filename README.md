# GlassesOnFace

This project aims to apply glasses' template over the face in the images. Firstly the face is detected in an image, then its orientation is detected and then using affine transformation, we transform the glasses' template and overlap it over the face image.


# Installation

Make sure that you have following libraries installed:
* os:           pip install os-sys
* cv2:          pip install opencv-python
* numpy:        pip install numpy
* matplotlib:   pip install matplotlib
* scipy:        pip install scipy
* scikit-image: pip install scikit-image
* dlib:         pip install dlib

### Installation steps:
* Clone this repository.
* Download [this file](https://drive.google.com/file/d/1ddB0ufdeH0-s5hMuvSTTEipkdrjYIK1Y/view?usp=sharing) and save it in the cloned folder(where main.py is present).
* Save the input face images in the folder `FaceImages`.
* Run the python code file `main.py`.
* See the results in `OutputImages` folder.


# 