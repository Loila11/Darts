from common import getClearImage, getEllipses, getFileName, toHSV, countDarts, writeSolution
from shapely.geometry import Point
import cv2


def getPolygons():
    """
    Get all areas of interest on the dartboard. In this case, it will be a list containing concentric discs for the
    flags, followed by a list of regions.

    :return: list of polygons
    """
    polygons = getEllipses('auxiliary_images/task2_template0.png', 100, 5000)
    for i in [1, 3]:
        polygons[i], polygons[i + 1] = polygons[i + 1], polygons[i]

    polygons += getEllipses('auxiliary_images/task2_template1.png', 500, 2000)
    polygons += getEllipses('auxiliary_images/task2_template2.png', 500, 2000)

    return polygons


def getPointScore(polygons, x, y, mapping_template):
    """
    Given a point's coordinates and the list of polygons, calculate the score at the given position.

    :param polygons: list of relevant regions on the board
    :param x: coordinate on the axis Ox
    :param y: coordinate on the axis Oy
    :param mapping_template: mapping between each polygon and the score inside it
    :return: the score at the given position
    """
    point = Point(x, y)

    if polygons[0].contains(point):
        return 'b50'
    if polygons[1].contains(point):
        return 'b25'
    for i in range(6, len(polygons)):
        if polygons[i].contains(point):
            score = 's'
            if polygons[3].contains(point) and not polygons[2].contains(point):
                score = 't'
            elif polygons[5].contains(point) and not polygons[4].contains(point):
                score = 'd'
            return score + str(mapping_template[i - 6])

    return '0'


def getTipPositions(darts):
    """
    Get the dart tips positions.

    :param darts: the list of darts
    :return: the list of positions where the darts enter the board
    """
    darts = [(int(dart[0][0] - 300), int(dart[0][1] + (dart[1][1] - dart[0][1]) / 2)) for dart in darts]
    return darts


def processImage(image, clearImage, image_name, polygons):
    """
    Process the given image, identify the darts on the board and write the solution.

    :param image: an image containing a dart board with some darts on it
    :param clearImage: the dartboard without darts, used as a mask
    :param image_name: image identifier, used for writing the output file
    :param polygons: the list of concentric discs and regions used for score calculation
    :return: None
    """
    diff = cv2.absdiff(image, clearImage)
    mask = toHSV(diff)
    darts = countDarts(mask, 12)

    mapping_template = [19, 17, 16, 15, 11, 6, 9, 4, 1, 5, 3, 7, 2, 8, 10, 14, 13, 18, 12, 20]
    writeSolution(
        'evaluation/Task2/' + image_name + '_predicted.txt',
        getTipPositions(darts),
        getPointScore,
        polygons,
        mapping_template
    )


def task2(path):
    """
    Get preliminary data and apply algorithm on all images.

    :param path: the path where to find the train / test data
    :return: None
    """
    # getClearImage('auxiliary_images/template_task2.jpg', 'auxiliary_images/task2_template0.png')

    polygons = getPolygons()

    clearImage = cv2.imread('auxiliary_images/template_task2.jpg')

    for i in range(1, 26):
        image_name = getFileName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, clearImage, image_name, polygons)
