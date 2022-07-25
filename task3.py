from common import getClearImage, getEllipses, getFileName, toHSV, countDarts, writeSolution, drawRectangle
from shapely.geometry import Point
import numpy as np
import cv2


def getPolygons(case):
    """
    Get all areas of interest on the dartboard. In this case, it will be a list containing concentric discs for the
    flags, followed by a list of regions.

    :param case: one of the 3 possible cases for the position of the board in a video. For each of these cases, we will
    return a different set of polygons
    :return: list of polygons
    """
    if case == 0:
        polygons = getEllipses('auxiliary_images/task3_template00.png', 100, 3000)
        polygons += getEllipses('auxiliary_images/task3_template01.png', 1000, 1500)
        polygons += getEllipses('auxiliary_images/task3_template02.png', 1000, 1500)
    elif case == 1:
        polygons = getEllipses('auxiliary_images/task3_template10.png', 100, 5000)
        polygons[0], polygons[1] = polygons[1], polygons[0]
        polygons += getEllipses('auxiliary_images/task3_template11.png', 500, 1500)
        polygons += getEllipses('auxiliary_images/task3_template12.png', 500, 1500)
    else:
        polygons = getEllipses('auxiliary_images/task3_template20.png', 100, 5000)
        polygons[0], polygons[1] = polygons[1], polygons[0]
        polygons += getEllipses('auxiliary_images/task3_template21.png', 700, 1500)
        polygons += getEllipses('auxiliary_images/task3_template22.png', 700, 1500)

    return polygons


def setData():
    """
    Set preliminary data as lists dependent on the video case.

    :return: The list of polygons, mappings between polygons and scores and template images for case identification
    """
    polygons = [
        getPolygons(0),
        getPolygons(1),
        getPolygons(2)
    ]
    mapping_template = [
        [18, 20, 1, 5],
        [19, 17, 16, 15, 9, 11, 3, 2, 7, 10, 8],
        [6, 11, 4, 9, 1, 5, 13, 14, 18, 12, 20]
    ]
    aux_image = [
        cv2.cvtColor(cv2.imread('auxiliary_images/template0_task3.jpg'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('auxiliary_images/template1_task3.jpg'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('auxiliary_images/template2_task3.jpg'), cv2.COLOR_BGR2GRAY)
    ]

    return polygons, mapping_template, aux_image


def getSimilarity(image1, image2):
    """
    Get similarity score between 2 images.

    :param image1: the first image
    :param image2: the second image
    :return: similarity score between images
    """
    score = cv2.matchTemplate(image1, image2, cv2.TM_SQDIFF_NORMED)
    min_val, _, _, _ = cv2.minMaxLoc(score)
    return min_val


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
    for i in range(4, len(polygons)):
        if polygons[i].contains(point):
            score = 's'
            if polygons[2].contains(point):
                score = 't'
            elif polygons[3].contains(point):
                score = 'd'
            return score + str(mapping_template[i - 4])

    return '0'


def processVideo(first_image, last_image, case, video_name, polygons, mapping_template):
    """
    Process the given image, identify the darts on the board and write the solution.

    :param first_image: the first frame of the video, used as a mask
    :param last_image: the last frame of the video, containing the dartboard with the last thrown dart
    :param case: one of the 3 possible cases for the position of the board in a video
    :param video_name: video identifier, used for writing the output file
    :param polygons: the list of concentric discs and regions used for score calculation
    :param mapping_template: mapping between each polygon and the score inside it
    :return: None
    """
    diff = cv2.absdiff(first_image, last_image)

    if case == 0:
        diff = diff[:, :600]

    diff = cv2.inRange(diff, np.array([20, 20, 30]), np.array([200, 200, 200]))
    darts = countDarts(diff, 5, 50, 20, 50)[-1]

    writeSolution(
        'evaluation/Task3/' + video_name + '_predicted.txt',
        [(darts[0][1], darts[1][0])],
        getPointScore,
        polygons,
        mapping_template,
        False
    )


def task3(path):
    """
    Get preliminary data and apply algorithm on the first and last frame of all videos.

    :param path: the path where to find the train / test data
    :return: None
    """
    # getClearImage('auxiliary_images/task3_template0.jpg', 'auxiliary_images/task3_template0.png')
    # getClearImage('auxiliary_images/task3_template1.jpg', 'auxiliary_images/task3_template1.png')
    # getClearImage('auxiliary_images/task3_template2.jpg', 'auxiliary_images/task3_template2.png')

    polygons, mapping_templates, aux_images = setData()

    for i in range(1, 26):
        video_name = getFileName(i)
        vidcap = cv2.VideoCapture(path + video_name + '.mp4')

        success, first_image = vidcap.read()
        last_image = aux_image = first_image
        while success:
            last_image = aux_image
            success, aux_image = vidcap.read()

        gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        case = (0, getSimilarity(gray, aux_images[0]))
        for j in [1, 2]:
            score = getSimilarity(gray, aux_images[j])
            if score < case[1]:
                case = (j, score)

        processVideo(first_image, last_image, case[0], video_name, polygons[case[0]], mapping_templates[case[0]])
