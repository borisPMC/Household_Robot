import cv2
import requests
import base64

import os
import time


file_path = "img.png"

img  =cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)



while 1:
    cv2.imshow("capture", img)
    keyvalue = cv2.waitKey(1)

    if keyvalue == 27:
        break
    elif keyvalue == ord('n'):
        break