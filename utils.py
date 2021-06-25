import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import datetime
def openCoordinateFindWindow(*filenames, color=(0,0,255), log=True):
    def callback(event, x, y, flags, param):
        if event==cv.EVENT_LBUTTONDOWN:
            print("mouse click at x: "+ str(x) +" and y: "+ str(y))
            cv.circle(image, (x, y), 5, color, thickness=6)
            cv.imshow(filename, image)
            x_list.append(x)
            y_list.append(y)
    
    x_list = []
    y_list = []
    
    for filename in filenames:
        image = cv.imread(filename)
        cv.imshow(filename, image)
        cv.setMouseCallback(filename, callback)


        cv.waitKey(0)
        cv.destroyAllWindows()

    x_range = [min(x_list), max(x_list)]
    y_range = [min(y_list), max(y_list)]

    print("Range of X: " + str(x_range))
    print("Range of Y: " + str(y_range))

    if log:
        with open("discover.log", "a") as logFile:
            logFile.write('Finding position boundaries; User Analysis: {} at {}\n'.format(str(filenames),str(datetime.datetime.now())))
            logFile.write('Range of X: {}\n'.format(str(x_range)))
            logFile.write('Range of Y: {}\n\n'.format(str(y_range)))
    return x_range, y_range

def cropImage(img, x_range, y_range):
    return img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

def discoverMask(filename, log=True):
    cv.namedWindow(filename)
    hl=hu=sl=su=vl=vu = -1
    def on_Trackbar(_):
        nonlocal hl, hu, sl, su, vl, vu
        hl = cv.getTrackbarPos("hue lower", filename)
        hu = cv.getTrackbarPos("hue upper", filename)

        sl = cv.getTrackbarPos("sat lower", filename)
        su = cv.getTrackbarPos("sat upper", filename)
        
        vl = cv.getTrackbarPos("val lower", filename)
        vu = cv.getTrackbarPos("val upper", filename)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (hl, sl, vl), (hu, su, vu))
    
        result = cv.bitwise_and(hsv, hsv, mask=mask)
        result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        cv.imshow(filename, result)

    cv.createTrackbar("hue lower", filename, 0, 255, on_Trackbar)
    cv.createTrackbar("hue upper", filename, 255, 255, on_Trackbar)

    cv.createTrackbar("sat lower", filename, 0, 255, on_Trackbar)
    cv.createTrackbar("sat upper", filename, 255, 255, on_Trackbar)

    cv.createTrackbar("val lower", filename, 0, 255, on_Trackbar)
    cv.createTrackbar("val upper", filename, 255, 255, on_Trackbar)

    img = cv.imread(filename)
    cv.imshow(filename, img)

    cv.waitKey(0)
    if log:
        with open("discover.log", "a") as logFile:
            logFile.write('Finding HSV boundaries; User Analysis: {} at {}\n'.format(str(filename),str(datetime.datetime.now())))
            logFile.write('Range of Hue: {}\n'.format(str((hl,hu))))
            logFile.write('Range of Sat: {}\n'.format(str((sl,su))))
            logFile.write('Range of Val: {}\n'.format(str((vl,vu))))
    cv.destroyAllWindows()

def discoverCanny(filename, log=True):
    img = cv.imread(filename)
    cv.namedWindow(filename)
    lower=upper=sobel=l2grad= -1
    def on_Trackbar(_):
        nonlocal lower, upper, sobel, l2grad
        lower = cv.getTrackbarPos("lower thres", filename)
        upper = cv.getTrackbarPos("upper thres", filename)

        sobel = cv.getTrackbarPos("sobel size", filename)
        l2grad = cv.getTrackbarPos("l2grad?", filename)

        canny = cv.Canny(img, lower, upper, apertureSize=sobel, L2gradient=bool(l2grad))
        cv.imshow(filename, canny)

    cv.createTrackbar("lower thres", filename, 100, 5000, on_Trackbar)
    cv.createTrackbar("upper thres", filename, 200, 5000, on_Trackbar)

    cv.createTrackbar("sobel size", filename, 3, 7, on_Trackbar)
    cv.createTrackbar("l2grad?", filename, 0, 1, on_Trackbar)

    cv.imshow(filename, img)
    cv.waitKey(0)

    if log:
        with open("discover.log", "a") as logFile:
            logFile.write('Finding Canny Edge values; User Analysis: {} at {}\n'.format(str(filename),str(datetime.datetime.now())))
            logFile.write('Range of Threshold: {}\n'.format(str((lower,upper))))
            logFile.write('Aperature/Sobel Size: {}\n'.format(str(sobel)))
            logFile.write('l2grad?: {}\n'.format(bool(l2grad)))
    cv.destroyAllWindows()

# in kwarg : descriptor -> default, upper
def discover(function, filename, preprocess=lambda x:x, **kwargs):
    img = cv.imread(filename)
    # preprocess the image (use this if function requires image requires special processesing before analyzing)
    img = preprocess(img)
    cv.namedWindow(filename)

    def on_Trackbar(_):
        for key in kwargs:
            kwargs[key][0] = cv.getTrackbarPos(key, filename)
        display = function(img, kwargs)
        cv.imshow(filename, display)

    for key in kwargs:
        cv.createTrackbar(key, filename, kwargs[key][0], kwargs[key][1], on_Trackbar)

    cv.imshow(filename, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def discoverHoughCircles(filename, gradient=cv.HOUGH_GRADIENT, log=True):
    kwargs = {}
    kwargs["dp (%)"] = [150, 10000]
    kwargs["minDist"] = [30, 10000]
    kwargs["param1"] = [10, 500]
    kwargs["param2 (%)"] = [100, 100]
    kwargs["minRadius"] = [40, 60]
    kwargs["maxRadius"] = [60, 500]
    def preprocess(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    def function(img, kwargs):
        circles = cv.HoughCircles(img, gradient, kwargs["dp (%)"][0]/100, kwargs["minDist"][0], kwargs["param1"][0], kwargs["param2 (%)"][0]/100, kwargs["minRadius"][0], kwargs["maxRadius"][0])
        circles = np.uint16(np.around(circles))
        print(np.shape(circles))
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
            # draw the center of the circle
            cv.circle(img,(i[0],i[1]),2,(0,0,255),5)
        return img
    discover(function, filename, preprocess=preprocess, **kwargs)


def useHSVMask(img, lower, upper):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, lower, upper)

    result = cv.bitwise_and(img_hsv, img_hsv, mask=mask)
    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    return result

if __name__ == "__main__":
    discoverMask("images/bloodweb_3_cropped.jpg")
