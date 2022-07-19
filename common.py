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


def drawRectangle(mask, point1, point2):
    clone = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
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

    return len(best_best_squares)


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
