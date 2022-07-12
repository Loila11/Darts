import numpy as np
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from matplotlib import pyplot as plt


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


def getDiff(path):
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    # aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2HSV)
    aux_gray = cv2.cvtColor(aux_image, cv2.COLOR_BGR2GRAY)

    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(image, aux_image)
    # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # canvas = np.zeros_like(image, np.uint8)
    # canvas[diff > 1] = image[diff > 1]

    cv2.imshow('diff', diff)
    cv2.waitKey(0)

    # toHSV(image, diff)

    # # start clearImage
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    #
    # # Downsize image (by factor 4) to speed up morphological operations
    # gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)
    #
    # # Morphological opening: Get rid of the stuff at the top of the ellipse
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #
    # # Resize image to original size
    # gray = cv2.resize(gray, dsize=(image.shape[1], image.shape[0]))
    # # end clearImage
    #
    # cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # print([len(point) for point in cnts])
    #
    # image = cv2.drawContours(cv2.imread(path + image_name), [polygon for polygon in cnts if len(cnts) > 1000], -1,
    #                          (0, 0, 255), 2)
    # out_image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
    # cv2.imwrite('test_diff_clear.png', out_image)

    # cv2.imwrite('test_diff_gray.png', gray)
    # cv2.imwrite('evaluation/Task1/.' + image_name, gray)


def toHSV(image, diff):
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    # green mask
    mask = cv2.inRange(diff, np.array([70, 50, 70]), np.array([128, 255, 255]))
    res = cv2.bitwise_and(image, image, mask=mask)

    # create resizable windows for displaying the images
    # cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    #
    # # display the images
    # cv2.imshow("mask", mask)
    # cv2.imshow("hsv", diff)
    # cv2.imshow("res", res)
    #
    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()

    return mask


def countDarts(path):
    image = cv2.imread(path)
    # image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)
    mask = toHSV(image, image)

    white = np.count_nonzero(np.all(mask == [255, 255, 255], axis=2))

    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)


def template_matching(path):
    template = cv2.imread('auxiliary_images/dart3.jpg', 0)
    template = cv2.resize(template, dsize=(0, 0), fx=0.15, fy=0.15)
    w, h = template.shape[::-1]

    image = cv2.imread(path, 0)

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = image.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def findArrow(path):
    pass


def task1(path):
    # getClearImage('auxiliary_images/template_task1.jpg', 'auxiliary_images/gray_removed_noise.png')
    polygons = getEllipses()
    path += '/Task1/'

    for i in range(1, 2):
        image_name = path
        if i < 10:
            image_name = '0'
        image_name += str(i) + '.jpg'

        # countDarts(image_name)
        # template_matching(image_name)
        # getDiff(image_name)
