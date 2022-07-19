from common import toHSV, countDarts, getImageName, getImageDiff
import cv2

outputs = [2, 1, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 3, 3, 2, 1]


def processImage(image, clearImage, image_name, i):
    image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)

    diff = cv2.absdiff(image, clearImage)
    mask = toHSV(diff)

    dartsNo = countDarts(mask, 20)
    if dartsNo != outputs[i - 1]:
        print(i, dartsNo, outputs[i - 1])

    # f = open('evaluation/Task2/' + image_name + '_predicted.txt', 'w')
    # f.write(str(dartsNo))
    # f.close()


def task2(path):
    # getClearImage('auxiliary_images/template_task2.jpg', 'auxiliary_images/gray_removed_noise.png')
    auxImage = cv2.imread('auxiliary_images/template_task2.jpg')
    clearImage = cv2.resize(auxImage, dsize=(0, 0), fx=0.2, fy=0.2)
    # clearImage = cv2.cvtColor(auxImage, cv2.COLOR_BGR2HSV)
    # clearImage = toHSV(auxImage, auxImage)

    path += '/Task2/'

    for i in range(1, 26):
        image_name = getImageName(i)
        image = cv2.imread(path + image_name + '.jpg')

        # processImage(image, clearImage, image_name, i)

        getImageDiff(image, auxImage)

        # template_matching(image_name)
        # getDiff(image_name)
