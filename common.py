import numpy as np
import cv2

from matplotlib import pyplot as plt
from IPython import display

from shapely.geometry.polygon import Polygon


def getImageName(i):
    """
    Compose image name from its number.

    :param i: image number
    :return: image name
    """
    image_name = ''
    if i < 10:
        image_name += '0'
    image_name += str(i)

    return image_name


def drawRectangle(image, point1, point2):
    """
    Draw a rectangle at the given position.
    code source: https://pythonmana.com/2022/03/202203090300273871.html

    :param image: image on which to draw
    :param point1: top-left point
    :param point2: bottom-right point
    :return: None
    """
    clone = image.copy()
    cv2.rectangle(clone, point1, point2, (0, 255, 0), 10)
    clone = clone[:, :, ::-1]
    plt.imshow(clone)
    plt.pause(0.1)
    display.clear_output(wait=True)


def getEllipses(path, th_low, th_up):
    """
    Get contours for the relevant data in an image (usually ellipses).

    :param path: the path where to find the image
    :param th_low: lower threshold for accepted polygon length
    :param th_up: upper threshold for accepted polygon length
    :return: a list of polygons
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = [polygon for polygon in cnts if th_low < len(polygon) < th_up]

    # Draw found contours in input image - task 1
    # print(len(cnts))
    # out_image = cv2.drawContours(cv2.imread('train/Task1/01.jpg'), cnts, -1, (0, 0, 255), 2)
    # out_image = cv2.resize(out_image, dsize=(0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('test_ellipses', out_image)
    # cv2.waitKey(0)

    # Draw found contours in input image - task 2
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


def getClearImage(origin, destination):
    """
    Preprocessing step - turn a template into its clear form, without noise or numbers. The clear image will be used to
    get the relevant polygons.
    code source: https://datascience.stackexchange.com/questions/69397/ellipses-detecting-at-the-image

    :param origin: original image file path
    :param destination: where to save the clear image version
    :return: None
    """
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


def toHSV(image):
    """
    Get the HSV version of an image and add a green mask.

    :param image: image to edit
    :return: green mask resulted from the image
    """
    diff = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # green mask
    mask = cv2.inRange(diff, np.array([70, 50, 70]), np.array([128, 255, 255]))

    # gray mask
    # mask = cv2.inRange(image, np.array([98, 94, 90]), np.array([188, 184, 183]))

    return mask


def getDartsAreas(best_squares):
    """
    Get the minimum number of rectangles containing relevant information by reuniting intersecting rectangles.

    :param best_squares: list of rectangles with relevant information
    :return: a list with the reunion of rectangles
    """
    best_best_squares = []
    for square in best_squares:
        overlaps = False
        for i in range(len(best_best_squares)):
            best_square = best_best_squares[i]
            if square[0][0] >= best_square[1][0] or \
               square[1][0] <= best_square[0][0] or \
               square[1][1] <= best_square[0][1] or \
               square[0][1] >= best_square[1][1]:
                continue

            best_best_squares[i] = ((min(square[0][0], best_square[0][0]), min(square[0][1], best_square[0][1])),
                                    (max(square[1][0], best_square[1][0]), max(square[1][1], best_square[1][1])),
                                    min(square[2], best_square[2]))
            overlaps = True

        if not overlaps:
            best_best_squares += [square]

    return best_best_squares


def countDarts(mask, score_th):
    """
    Find the flags for all darts on board. In order to do this, use a sliding window that finds the squares with the
    percentage of relevant information smaller than the threshold, add them to a list and reduce the list by joining the
    squares that intersect.

    :param mask: mask applied on the original image that contains relevant information
    :param score_th: maximum score threshold
    :return: list of flag positions
    """
    best_squares = []
    size = 100
    step = 50
    for y in range(250, mask.shape[0] - size, step):
        for x in range(250, mask.shape[1] - size, step):
            window = mask[y:y + size, x:x + size]
            white = np.count_nonzero(window)
            score = 100000 if white == 0 else size * size / white
            if score < score_th:
                best_squares += [((x, y), (x + size, y + size), score)]

    best_best_squares = getDartsAreas(best_squares)
    best_best_squares = getDartsAreas(best_best_squares)

    return best_best_squares


def getBestSimilarity(template_path, path):
    """
    Use pattern matching to correctly position an image with darts on top of the original template.

    :param template_path: path for the template file
    :param path: path for the image file
    :return: absolute difference between the final overlapping
    """
    template = cv2.imread(template_path, 0)
    template = template[50:-50, 50:-50]
    w, h = template.shape[::-1]

    image = cv2.imread(path, 0)

    score = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score)

    top_left = min_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    image = cv2.GaussianBlur(image, (5, 5), 0)

    diff = cv2.absdiff(image, template)

    cv2.imshow('diff', diff)
    cv2.waitKey(0)

    return diff


def writeSolution(path, darts, getPointScore, polygons):
    """
    Get solution and write it in an output file.

    :param path: output file path
    :param darts: list of dart flag positions
    :param getPointScore: function used to calculate the score at a given position - different for task 2 and 3
    :param polygons: list of polygons with relevant information
    :return: None
    """
    f = open(path, 'w')
    f.write(str(len(darts)))

    for dart in darts:
        x = int(dart[0][0] - 300)
        y = int(dart[0][1] + (dart[1][1] - dart[0][1]) / 2)
        score = getPointScore(polygons, x, y)
        f.write('\n' + str(score))

    f.close()
