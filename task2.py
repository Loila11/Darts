from common import toHSV, countDarts, getImageName, getImageDiff, drawRectangle
import cv2

from matplotlib import pyplot as plt
from IPython import display

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def getEllipses(path, th_low, th_high):
    image = cv2.imread(path)
    # image = cv2.bitwise_not(image)
    # cv2.imwrite('auxiliary_images/task2_template2.png', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # diff = cv2.resize(diff, dsize=(0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)

    # Find contours
    cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = [polygon for polygon in cnts if th_low < len(polygon) < th_high]

    # Draw found contours in input image
    # print(len(cnts))
    # for i in range(len(cnts)):
    #     print(len(cnts[i]))
    #     out_image = cv2.drawContours(image, cnts[i], -1, (0, 0, 255), 10)
    #     # out_image = cv2.resize(out_image, dsize=(0, 0), fx=0.2, fy=0.2)
    #     plt.imshow(out_image)
    #     plt.pause(0.1)
    #     display.clear_output(wait=True)

    polygons = [Polygon([(point[0], point[1]) for [point] in polygon]) for polygon in cnts]
    return polygons


def getPointScore(polygons, x, y):
    mapping_template = [19, 17, 16, 15, 11, 6, 9, 4, 1, 5, 3, 7, 2, 8, 10, 14, 13, 18, 12, 20]
    point = Point(x, y)

    if polygons[0].contains(point):
        return 'b50'
    if polygons[1].contains(point):
        return 'b25'
    for i in range(2, len(polygons)):
        if polygons[i].contains(point):
            return 's' + str(mapping_template[i - 2])

    return '0'


def processImage(image, clearImage, image_name, polygons):
    diff = cv2.absdiff(image, clearImage)
    mask = toHSV(diff)

    darts = countDarts(mask, 12)

    f = open('evaluation/Task2/' + image_name + '_predicted.txt', 'w')
    f.write(str(len(darts)))

    for dart in darts:
        x = int(dart[0][0] - 300)
        y = int(dart[0][1] + (dart[1][1] - dart[0][1]) / 2)
        score = getPointScore(polygons, x, y)
        f.write('\n' + str(score))

    f.close()


def task2(path):
    # getClearImage('auxiliary_images/template_task2.jpg', 'auxiliary_images/gray_removed_noise2.png')
    clearImage = cv2.imread('auxiliary_images/template_task2.jpg')
    polygons = getEllipses('auxiliary_images/task2_template2.png', 100, 500)
    polygons += getEllipses('auxiliary_images/task2_template1.png', 500, 2000)
    polygons += getEllipses('auxiliary_images/task2_template2.png', 500, 2000)
    path += '/Task2/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, clearImage, image_name, polygons)
        # getImageDiff(image, auxImage)
        # template_matching(image_name)
        # getDiff(image_name)
