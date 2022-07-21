import numpy as np
import cv2

from matplotlib import pyplot as plt
from IPython import display


def getImageName(i):
    image_name = ''
    if i < 10:
        image_name += '0'
    image_name += str(i)

    return image_name


def drawRectangle(image, point1, point2):
    clone = image.copy()
    cv2.rectangle(clone, point1, point2, (0, 255, 0), 2)
    clone = clone[:, :, ::-1]
    plt.imshow(clone)
    plt.pause(0.1)
    display.clear_output(wait=True)


def toHSV(diff):
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    # green mask
    mask = cv2.inRange(diff, np.array([70, 50, 70]), np.array([128, 255, 255]))
    # res = cv2.bitwise_and(diff, diff, mask=mask)

    # create resizable windows for displaying the images
    # cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    # display the images
    # cv2.imshow("mask", mask)
    # cv2.imshow("hsv", diff)
    # cv2.imshow("res", res)

    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()

    return mask


def getDartsAreas(best_squares):
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
    best_squares = []
    size = 30
    step = 10
    for y in range(50, mask.shape[0] - size, step):
        for x in range(50, mask.shape[1] - size, step):
            window = mask[y:y + size, x:x + size]
            white = np.count_nonzero(window)
            score = 100000 if white == 0 else size * size / white
            if score < score_th:
                best_squares += [((x, y), (x + size, y + size), score)]

    best_best_squares = getDartsAreas(best_squares)

    return best_best_squares


def getImageDiff(image, aux_image):
    diff = cv2.absdiff(image, aux_image)
    diff = cv2.resize(diff, dsize=(0, 0), fx=0.2, fy=0.2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 100, 200)

    cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

    cv2.imshow('diff', diff)
    cv2.imshow('gray', gray)
    cv2.imshow('edges', edges)

    cv2.waitKey(0)


def getBestSimilarity(template_path, path):
    template = cv2.imread(template_path, 0)
    template = template[50:-50, 50:-50]
    template = cv2.resize(template, dsize=(0, 0), fx=0.2, fy=0.2)
    w, h = template.shape[::-1]

    image = cv2.imread(path, 0)
    image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)

    score = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score)

    top_left = min_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # template = cv2.GaussianBlur(template, (5, 5), 0)

    diff = cv2.absdiff(image, template)

    # mask = cv2.inRange(image, np.array([98, 94, 90]), np.array([188, 184, 183]))
    # diff = cv2.bitwise_and(template, template, mask=image)
    # diff = cv2.bitwise_or(image, image, mask=template)
    # diff = cv2.bitwise_not(template)

    # edges = cv2.Canny(diff, 100, 200)
    # diff = cv2.resize(diff, dsize=(0, 0), fx=0.2, fy=0.2)

    cv2.imshow('diff', diff)
    cv2.waitKey(0)

    return diff

    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         top_left = min_loc
    #     else:
    #         top_left = max_loc


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
