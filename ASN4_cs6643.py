import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils

cap = cv2.VideoCapture('/Users/admin/Downloads/ASN4/Vid.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
r1 = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
while (True):
    ret, frame = cap.read()
    if ret:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('/Users/admin/Downloads/Vid_042.jpg', 0)
        w, h = template.shape[::-1]
        # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.8
        # loc = np.where(res >= threshold)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), 255, 2)

        found = None
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))
            r = img_gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            r = img_gray.shape[1] / float(resized.shape[1])
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            found = (maxVal, maxLoc, r)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
        cv2.rectangle(frame, (startX, startY), (endX, endY), 255, 2)
        r1.write(frame)
        cv2.imshow("res", frame)
    else:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
r1.release()
