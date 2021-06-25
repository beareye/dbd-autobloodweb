import cv2 as cv
import numpy as np
import utils

img = cv.imread("./images/bloodweb_4.jpg")
img = utils.useHSVMask(img, (0, 0, 0), (90, 90, 255))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = utils.cropImage(img, (0, len(img[0])//3*2), (0, len(img)))
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,param1=.1,param2=50, minRadius=35, maxRadius=55)
circles = np.uint16(np.around(circles[0,:]))
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for i in circles:
    print(circles)
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
    # draw the center of the circle
    cv.putText(img, str(i[2]), (i[0],i[1]), cv.FONT_HERSHEY_SIMPLEX, 1, cv.QT_FONT_BLACK, thickness=3)
print(len(circles))
#cv.imshow("a", img_grey)
cv.imshow("b", img)
cv.waitKey(0)
def main():
    pass

if __name__ == "__main__":
    main()
