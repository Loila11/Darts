from common import toHSV, countDarts, getImageName, getImageDiff, drawRectangle, getEllipses, writeSolution
from shapely.geometry import Point
import cv2


def getPolygons():
    polygons = getEllipses('auxiliary_images/task2_template2.png', 100, 500)
    polygons += getEllipses('auxiliary_images/task2_template1.png', 500, 2000)
    polygons += getEllipses('auxiliary_images/task2_template2.png', 500, 2000)
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

    writeSolution('evaluation/Task2/' + image_name + '_predicted.txt', darts, getPointScore, polygons)


def task2(path):
    # getClearImage('auxiliary_images/template_task2.jpg', 'auxiliary_images/gray_removed_noise2.png')
    clearImage = cv2.imread('auxiliary_images/template_task2.jpg')
    polygons = getPolygons()
    path += '/Task2/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, clearImage, image_name, polygons)
        # getImageDiff(image, auxImage)
        # template_matching(image_name)
        # getDiff(image_name)
