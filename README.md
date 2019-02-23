# face_recognition_system
Implementation of Face Recognition in the Wild

# Intro
This project is to investigate different face recognition technology in the wild. Since some of the algorithms are experimented exclusively with a controlled-environment database, it performs poorly in unconstrained datasets. For this project, multiple of algorithms are examined and experimented with the constrained environment database comparatively.

# Methods
Image pre-processing techniques are performed first. It includes face detection, normalization with image size, rotation and cropping, illumination normalization.
Perform feature extraction on the images. For PCA/LDA, it forms a basis weight vector showing how an image can be represented by a linear combination. Classification is performed by nearest neighbor method.
For LBP, perform the operator to the image to extract the features. Classification is taken by calculating histogram similarity by weighted chi square statistic, log- likelihood statistic, histogram intersection.

# Results
The experiment is taken by cross-validation. It includes changing the number of training images, eigenfaces or fisherfaces for the above 3 methods. It is all experimented on 2 different databases.

## Usage 
###  Use cmake to build the system 
```
cmake .
```
