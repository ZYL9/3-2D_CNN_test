import spectral
import numpy as np
import cv2 as cv
import scipy.io as sio
import matplotlib.pyplot as plt

y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
outputs = np.loadtxt(open("./new.csv", "rb"), delimiter=",", skiprows=0)

# outputs *= 20
# image = np.asanyarray(outputs, dtype=np.uint8)
# cv.imshow('', image)
# cv.waitKey()

ground_truth=spectral.imshow(classes=y.astype(int),figsize=(9,9))
predice_image=spectral.imshow(classes=outputs.astype(int),figsize=(9,9))
plt.show()
