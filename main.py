import numpy as np
import cv2


def task3(path):
    pass


def task2(path):
    pass


def toHSV(image, diff):
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(diff, np.array([90, 50, 70]), np.array([128, 255, 255]))
    res = cv2.bitwise_and(image, image, mask=mask)

    # create resizable windows for displaying the images
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    # display the images
    cv2.imshow("mask", mask)
    cv2.imshow("hsv", diff)
    cv2.imshow("res", res)

    if cv2.waitKey(0):
        cv2.destroyAllWindows()


def diffOpenCV(path, image_name):
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    # aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2HSV)
    # aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2GRAY)

    image = cv2.imread(path + image_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(image, aux_image)
    # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # canvas = np.zeros_like(image, np.uint8)
    # canvas[diff > 1] = image[diff > 1]

    # toHSV(image, diff)

    cv2.imwrite('test_diff.png', diff)
    # cv2.imwrite('evaluation/Task1/' + image_name, diff)


def task1(path):
    # 'evaluation/Task1/...'
    path += '/Task1/'

    for i in range(1, 2):
        image_name = ''
        if i < 10:
            image_name = '0'
        image_name += str(i) + '.jpg'

        diffOpenCV(path, image_name)


def main(path):
    task1(path)


main('train')
