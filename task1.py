from common import toHSV, countDarts, getImageName, getImageDiff, getBestSimilarity
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

outputs = [3, 2, 3, 2, 1, 2, 1, 1, 2, 2, 3, 1, 3, 3, 2, 1, 3, 3, 1, 3, 1, 2, 2, 2, 1]


def getClearImage(origin, destination):
    image = cv2.imread(origin)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Downsize image (by factor 4) to speed up morphological operations
    gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)

    # Morphological opening: Get rid of the stuff at the top of the ellipse
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Resize image to original size
    gray = cv2.resize(gray, dsize=(image.shape[1], image.shape[0]))
    cv2.imwrite(destination, gray)

    return gray


def getEllipses():
    image = cv2.imread('auxiliary_images/gray_removed_noise.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = [polygon for polygon in cnts if 200 < len(polygon) < 10000]

    # Draw found contours in input image
    image = cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    # print(len(cnts))

    # Downsize image
    out_image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
    cv2.imwrite('test_ellipses.png', out_image)

    polygons = [Polygon([(point[0], point[1]) for [point] in polygon]) for polygon in cnts]
    for i in [1, 3, 5, 7]:
        polygons[i], polygons[i + 1] = polygons[i + 1], polygons[i]

    return polygons


def getPointScore(polygons, x, y):
    point = Point(x, y)
    for i in range(9, -1, -1):
        if polygons[i].contains(point):
            return 10 - i
    return 0


def processImage(image, image_name, i):
    image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)
    mask = toHSV(image)

    dartsNo = countDarts(mask, 40)
    if dartsNo != outputs[i - 1]:
        print(i, dartsNo, outputs[i - 1])

    f = open('evaluation/Task1/' + image_name + '_predicted.txt', 'w')
    f.write(str(dartsNo))
    f.close()


def task1(path):
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/gray_removed_noise.png')
    # polygons = getEllipses()
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    path += '/Task1/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        # processImage(image, image_name, i)
        # getImageDiff(image, aux_image)
        getBestSimilarity(path + image_name + '.jpg', 'auxiliary_images/template_task1.jpg')

        # template_matching(image_name)
        # getDiff(image_name)
