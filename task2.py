import numpy as np
import cv2


def toHSV(image, diff):
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    # green mask
    mask = cv2.inRange(diff, np.array([70, 50, 70]), np.array([128, 255, 255]))
    # res = cv2.bitwise_and(image, image, mask=mask)

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
    # image = cv2.imread(path)
    # image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)
    #
    # diff = cv2.absdiff(image, auxMask)
    # mask = toHSV(diff, diff)

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


def task2(path):
    # getClearImage('auxiliary_images/template_task2.jpg', 'auxiliary_images/gray_removed_noise.png')
    auxImage = cv2.imread('auxiliary_images/template_task2.jpg')
    clearImage = cv2.resize(auxImage, dsize=(0, 0), fx=0.2, fy=0.2)
    # clearImage = cv2.cvtColor(auxImage, cv2.COLOR_BGR2HSV)
    # clearImage = toHSV(auxImage, auxImage)

    path += '/Task2/'
    outputs = [2, 1, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 3, 3, 2, 1]

    for i in range(1, 26):
        image_name = ''
        if i < 10:
            image_name += '0'
        image_name += str(i)

        image = cv2.imread(path + image_name + '.jpg')
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)

        diff = cv2.absdiff(image, clearImage)
        mask = toHSV(diff, diff)
        dartsNo = countDarts(mask, 20)

        # dartsNo = countDarts(path + image_name + '.jpg', clearImage)
        if dartsNo != outputs[i - 1]:
            print(i, dartsNo, outputs[i - 1])

        # f = open('evaluation/Task1/' + image_name + '_predicted.txt', 'w')
        # f.write(str(dartsNo))
        # f.close()

        # template_matching(image_name)
        # getDiff(image_name)
