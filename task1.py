from common import toHSV, countDarts, getImageName, getImageDiff, getBestSimilarity, getEllipses, writeSolution
from shapely.geometry import Point
import cv2


def getPolygons():
    polygons = getEllipses('auxiliary_images/gray_removed_noise.png', 200, 10000)
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

    writeSolution('evaluation/Task1/' + image_name + '_predicted.txt', darts, getPointScore, polygons)


def task1(path):
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/gray_removed_noise.png')
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    polygons = getPolygons()
    path += '/Task1/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, image_name, polygons)
        # getImageDiff(image, aux_image)
        # getBestSimilarity(path + image_name + '.jpg', 'auxiliary_images/template_task1.jpg')
        # template_matching(image_name)
        # getDiff(image_name)
