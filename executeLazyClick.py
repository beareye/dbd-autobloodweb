import cv2 as cv
import pyautogui as pag
import numpy as np
import utils

from time import sleep
from random import choice
from keyboard import on_press_key
from os import _exit
from operator import sub

radius = 45

def main():
    pag.keyDown('alt')
    pag.press('tab')
    pag.keyUp('alt')
    on_press_key("escape", lambda _: _exit(0))  
    while True:
        spendPointsOnPage()

def spendPointsOnPage():
    items = findItemCoordinates()
    while True: # Loop until there are no more circles
        circles = findCircles()
        if circles.size == 0:
            pag.click()
            sleep(1)
            return
        nextCircle = getBestCircle(circles, items)
        if nextCircle is []:
            return

        pag.moveTo(*nextCircle)
        pag.mouseDown()
        sleep(.5)
        pag.mouseUp()

def findCircles():
    pag.moveTo(10,10) # move mouse to a better position to remove hindrances
    pag.click()
    screenshot = pag.screenshot()
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    screenshot = utils.cropImage(screenshot, (0, len(screenshot[0])//3*2), (0, len(screenshot))) # get left side of screen
    screenshot = utils.useHSVMask(screenshot, (0, 0, 0), (90, 90, 255)) # Get white circles
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(screenshot, cv.HOUGH_GRADIENT, 1, 20,param1=.1,param2=50, minRadius=radius-10, maxRadius=radius+10)
    if circles is None:
        print("No circles")
        return np.empty((0,0))
    circles = np.around(circles[0,:])
    circles = np.uint16(circles)
    return circles

def getBestCircle(circles, items):
    if items == []:
        print("random")
        #items.extend(findItemCoordinates())
        return choice(circles)[0:2]
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
    return closestCircle

def findItemCoordinates(display=False):
    screenshot = pag.screenshot()
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2GRAY)
    img = utils.cropImage(screenshot, (0, len(screenshot[0])//3*2), (0, len(screenshot))) # get left side of screen
    itemsList = ["hand", "purple", "eventcake",  "streamers", "greenbattery", "green", "event", "yellowbattery", "yellow", "filament", "brownbattery", "alex", "yellowcoin"]
    threshold = .75
    coordinates = []
    for item in itemsList:
        print(item)
        itemImage = cv.imread("./images/{}.jpg".format(item), cv.IMREAD_GRAYSCALE)
        res = cv.matchTemplate(img, itemImage, cv.TM_CCOEFF_NORMED)
        coordinates += zip(*(np.where(res >= threshold)[::-1]))
    if display:
        for pt in coordinates:
            cv.circle(img, pt, radius, (0,0,255), thickness=3)
        cv.imshow("a", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    coordinates = [[coord+radius//2 for coord in point] for point in coordinates] # bring coordinates to center of item
    coordinates = cleanCoordinates(coordinates)

    # if display:
    #     for pt in coordinates:
    #         cv.circle(img, pt, radius, (0,0,255), thickness=3)
    #     cv.imshow("a", img)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    print("{} items found".format(len(coordinates)))
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
