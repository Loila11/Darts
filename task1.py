from common import toHSV, countDarts, getImageName, getImageDiff, getBestSimilarity, drawRectangle

import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def getEllipses():
    image = cv2.imread('auxiliary_images/gray_removed_noise.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = [polygon for polygon in cnts if 200 < len(polygon) < 10000]

    # Draw found contours in input image
    # print(len(cnts))
    # out_image = cv2.drawContours(cv2.imread('train/Task1/01.jpg'), cnts, -1, (0, 0, 255), 2)
    # out_image = cv2.resize(out_image, dsize=(0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('test_ellipses', out_image)
    # cv2.waitKey(0)

    polygons = [Polygon([(point[0], point[1]) for [point] in polygon]) for polygon in cnts]
    for i in [1, 3, 5, 7]:
        polygons[i], polygons[i + 1] = polygons[i + 1], polygons[i]

    return polygons


def getPointScore(polygons, x, y):
    point = Point(x, y)
    for i in range(10):
        if polygons[i].contains(point):
            return 10 - i
    return 0


def processImage(image, image_name, polygons):
    mask = toHSV(image)

    darts = countDarts(mask, 40)

    f = open('evaluation/Task1/' + image_name + '_predicted.txt', 'w')
    f.write(str(len(darts)))

    for dart in darts:
        x = int(dart[0][0] - 300)
        y = int(dart[0][1] + (dart[1][1] - dart[0][1]) / 2)
        score = getPointScore(polygons, x, y)
        f.write('\n' + str(score))

    f.close()


def task1(path):
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/gray_removed_noise.png')
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    polygons = getEllipses()
    path += '/Task1/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, image_name, polygons)
        # getImageDiff(image, aux_image)
        # getBestSimilarity(path + image_name + '.jpg', 'auxiliary_images/template_task1.jpg')
        # template_matching(image_name)
        # getDiff(image_name)
