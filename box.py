import cv2
import numpy as np
import time

# This variable determines if we want to load color range from memory
# or use the ones defined in the notebook.
load_from_disk = True

# If true then load color range from memory
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

# kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# set the window to auto-size so we can view this full screen.
cv2.namedWindow('Pen', cv2.WINDOW_NORMAL)


noiseth = 500

while (1):

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If you're reading from memory then load the upper and lower
    # ranges from there
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]


    else:
        lower_range = np.array([26, 80, 147])
        upper_range = np.array([81, 255, 255])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)


    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)


    if contours and cv2.contourArea(max(contours,
                                        key=cv2.contourArea)) > noiseth:
        # Grab the biggest contour with respect to area
        c = max(contours, key=cv2.contourArea)

        # Get bounding box coordinates around that contour
        x, y, w, h = cv2.boundingRect(c)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 25, 255), 2)

    cv2.imshow('image', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()