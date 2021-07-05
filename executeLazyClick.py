import cv2 as cv
import pyautogui as pag
import numpy as np
import utils
import time
import random
from operator import sub

radius = 45

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
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2GRAY)
    screenshot = utils.cropImage(screenshot, (0, len(screenshot[0])//3*2), (0, len(screenshot))) # get left side of screen
    items = findItemCoordinates(screenshot, display=False)
    while True:
        pag.moveTo(10,10)
        screenshot = pag.screenshot()
        screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        screenshot = utils.cropImage(screenshot, (0, len(screenshot[0])//3*2), (0, len(screenshot))) # get left side of screen
        screenshot = utils.useHSVMask(screenshot, (0, 0, 0), (90, 90, 255)) # Get white circles
        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        circles = cv.HoughCircles(screenshot, cv.HOUGH_GRADIENT, 1, 20,param1=.1,param2=50, minRadius=radius-10, maxRadius=radius+10)

        try:
            circles = np.around(circles[0,:])
        except:
            print("no circles")
            pag.click()
            time.sleep(2)
            return
        
        circles = np.uint16(circles)
        nextCircle = getBestCircle(circles, items)
        if nextCircle == []:
            return

        pag.moveTo(*nextCircle)
        pag.mouseDown()
        time.sleep(.5)
        pag.mouseUp()
        time.sleep(.1)

def getBestCircle(circles, items):
    if items == []:
        print("random")
        return random.choice(circles)[0:2]
    for circle in circles:
        for item in items:
            if np.linalg.norm(circle[0:2]-item) < radius:
                print("item")
                items.remove(item)
                return item
    
    minDist = 9999
    closestCircle = []
    for circle in circles:
        nextVal = np.linalg.norm(circle[0:2]-items[0])
        if nextVal < minDist:
            closestCircle = circle[0:2]
            minDist = nextVal
    print(minDist)
    return closestCircle


def findItemCoordinates(img, display=False):
    itemsList = ["purple", "streamers", "greenbattery", "green", "event", "yellowbattery", "yellow", "filament", "brownbattery"]
    threshold = .8
    coordinates = []
    for item in itemsList:
        itemImage = cv.imread("./images/{}.jpg".format(item), cv.IMREAD_GRAYSCALE)
        res = cv.matchTemplate(img, itemImage, cv.TM_CCOEFF_NORMED)
        coordinates += zip(*(np.where(res >= threshold)[::-1]))
        print("search {}".format(item))
    
    coordinates = [[coord+radius//2 for coord in point] for point in coordinates] # bring coordinates to center of item
    coordinates = cleanCoordinates(coordinates)
    
    if display:
        for pt in coordinates:
            cv.circle(img, pt, radius, (0,0,255), thickness=3)
        cv.imshow("a", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return coordinates

def cleanCoordinates(coordinates):
    newCoordinates = []
    for coordinate in coordinates:
        if len(newCoordinates) == 0:
            newCoordinates.append(coordinate)
        else:
            if np.all([np.linalg.norm(list(map(sub,newCoordinate,coordinate))) >= radius for newCoordinate in newCoordinates]):
                newCoordinates.append(coordinate)
    return newCoordinates

if __name__ == "__main__":
    main()
