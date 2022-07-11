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


def getClearImage():
    image = cv2.imread('auxiliary_images/template_task1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Downsize image (by factor 4) to speed up morphological operations
    gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)

    # Morphological opening: Get rid of the stuff at the top of the ellipse
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Resize image to original size
    gray = cv2.resize(gray, dsize=(image.shape[1], image.shape[0]))
    cv2.imwrite('Gray_removed_noise.png', gray)


def getEllipses():
    image = cv2.imread('Gray_removed_noise.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = [cnts[i] for i in range(len(cnts)) if 200 < len(cnts[i]) < 10000]

    # Draw found contours in input image
    image = cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    # print(len(cnts))

    # Downsize image
    out_image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
    cv2.imwrite('test_ellipses.png', out_image)

    return cnts


def diffOpenCV(path, image_name):
    aux_image = cv2.imread('auxiliary_images/template_task1.jpg')
    # aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2HSV)
    # aux_gray = cv2.cvtColor(aux_image, cv2.COLOR_BGR2GRAY)

    image = cv2.imread(path + image_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(image, aux_image)
    # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # canvas = np.zeros_like(image, np.uint8)
    # canvas[diff > 1] = image[diff > 1]

    # toHSV(image, diff)

    # cv2.imwrite('test_diff_gray.png', gray)
    # cv2.imwrite('evaluation/Task1/.' + image_name, gray)


def task1(path):
    # getClearImage()
    areas = getEllipses()
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
