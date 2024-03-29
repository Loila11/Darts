from common import getClearImage, getEllipses, getFileName, toHSV, countDarts, writeSolution
from shapely.geometry import Point
import cv2


def getPolygons():
    """
    Get all areas of interest on the dartboard. In this case, it will be a list of concentric discs, each disc
    representing a score on the board.

    :return: list of polygons
    """
    polygons = getEllipses('auxiliary_images/task1_template0.png', 200, 10000)
    for i in [1, 3, 5, 7]:
        polygons[i], polygons[i + 1] = polygons[i + 1], polygons[i]

    return polygons


def getPointScore(polygons, x, y, mapping_template):
    """
    Given a point's coordinates and the list of polygons, calculate the score at the given position.

    :param polygons: list of concentric discs
    :param x: coordinate on the axis Ox
    :param y: coordinate on the axis Oy
    :param mapping_template: mapping between each polygon and the score inside it
    :return: the score at the given position
    """
    point = Point(x, y)
    for i in range(10):
        if polygons[i].contains(point):
            return str(mapping_template[i])
    return '0'


def getTipPositions(darts):
    """
    Get the dart tips positions.

    :param darts: the list of darts
    :return: the list of positions where the darts enter the board
    """
    darts = [(int(dart[0][0] - 300), int(dart[0][1] + (dart[1][1] - dart[0][1]) / 2)) for dart in darts]
    return darts


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

    mapping_template = [10 - i for i in range(10)]
    writeSolution(
        'evaluation/Task1/' + image_name + '_predicted.txt',
        getTipPositions(darts),
        getPointScore,
        polygons,
        mapping_template
    )


def task1(path):
    """
    Get preliminary data and apply algorithm on all images.

    :param path: the path where to find the train / test data
    :return: None
    """
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/task1_template0.png')

    polygons = getPolygons()

    for i in range(1, 26):
        image_name = getFileName(i)
        image = cv2.imread(path + image_name + '.jpg')

        processImage(image, image_name, polygons)
