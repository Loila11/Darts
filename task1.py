from common import toHSV, countDarts, getImageName, getEllipses, writeSolution
from shapely.geometry import Point
import cv2


def getPolygons():
    """
    Get all areas of interest on the dartboard. In this case, it will be a list of concentric discs, each disc
    representing a score on the board.

    :return: list of polygons
    """
    polygons = getEllipses('auxiliary_images/gray_removed_noise.png', 200, 10000)
    for i in [1, 3, 5, 7]:
        polygons[i], polygons[i + 1] = polygons[i + 1], polygons[i]

    return polygons


def getPointScore(polygons, x, y):
    """
    Given a point's coordinates and the list of polygons, calculate the score at the given position.

    :param polygons: list of concentric discs
    :param x: coordinate on the axis Ox
    :param y: coordinate on the axis Oy
    :return: the score at the given position
    """
    point = Point(x, y)
    for i in range(10):
        if polygons[i].contains(point):
            return 10 - i
    return 0


def processImage(image, image_name, polygons):
    """
    Process the given image, identify the darts on the board and write the solution.

    :param image: an image containing a dart board with some darts on it
    :param image_name: image identifier, used for writing the output file
    :param polygons: the list of concentric discs used for score calculation
    :return: None
    """
    mask = toHSV(image)
    darts = countDarts(mask, 40)

    writeSolution('evaluation/Task1/' + image_name + '_predicted.txt', darts, getPointScore, polygons)


def task1(path):
    """
    Get preliminary data and apply algorithm on all images.

    :param path: the path where to find the train / test data
    :return: None
    """
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/gray_removed_noise.png')
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    polygons = getPolygons()

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, image_name, polygons)
