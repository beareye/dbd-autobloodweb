import cv2 as cv
import pyautogui as pag
import numpy as np
import utils
import time

def main():
    print("Starting: (make sure ur last tab open was dbd) works on both 1080p and 2k resolutions")
    pag.keyDown('alt')
    time.sleep(.1)
    pag.press('tab')
    pag.keyUp('alt')
    while True:
        spendPoints()
    

def spendPoints():
    screenshot = pag.screenshot()
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    screenshot = utils.cropImage(screenshot, (0, len(screenshot[0])//3*2), (0, len(screenshot)))
    screenshot = utils.useHSVMask(screenshot, (0, 0, 0), (90, 90, 255))
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(screenshot, cv.HOUGH_GRADIENT, 1, 20,param1=.1,param2=50, minRadius=35, maxRadius=55)
    try:
        circles = np.around(circles[0,:])
        circles = np.uint16(circles)
        if (len(circles) > 0):
            pag.moveTo(circles[0][0], circles[0][1])
            pag.mouseDown()
            time.sleep(.5)
            pag.mouseUp()
    except:
        pag.click()
        time.sleep(.25)
if __name__ == "__main__":
    main()