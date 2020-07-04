import cv2
import numpy as np
import time
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
pen_img = cv2.resize(cv2.imread('pen.png', 1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg', 1), (50, 50))
kernel = np.ones((5, 5), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

canvas = None


backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
background_threshold = 600

switch = 'Pen'
last_switch = time.time()
x1, y1 = 0, 0
noiseth = 800
wiper_thresh = 40000
clear = False

while (1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    if canvas is None:
        canvas = np.zeros_like(frame)

    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)
    switch_thresh = np.sum(fgmask == 255)


    if switch_thresh > background_threshold and (time.time() - last_switch) > 1:
        last_switch = time.time()
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
    else:
        lower_range = np.array([26, 80, 147])
        upper_range = np.array([81, 255, 255])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours,
                                        key=cv2.contourArea)) > noiseth:

        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            if switch == 'Pen':
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1, y1),
                                  (x2, y2), [255, 0, 0], 5)

            else:
                cv2.circle(canvas, (x2, y2), 20,
                           (0, 0, 0), -1)

        # After the line is drawn the new points become the previous points.
        x1, y1 = x2, y2
        if area > wiper_thresh:
            cv2.putText(canvas, 'Clearing Canvas', (0, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
            clear = True
    else:
        x1, y1 = 0, 0

    _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20,
                            255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
    background = cv2.bitwise_and(frame, frame,
                                 mask=cv2.bitwise_not(mask))
    frame = cv2.add(foreground, background)

    # Switch the images depending upon what we're using, pen or eraser.
    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255, 255, 255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img

    cv2.imshow('image', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    # Clear the canvas after 1 second, if the clear variable is true
    if clear == True:
        time.sleep(1)
        canvas = None
        clear = False

cv2.destroyAllWindows()
cap.release()