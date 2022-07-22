from task1 import task1
from task2 import task2

from common import getImageName


def checkScore(path):
    accuracy = 0

    for i in range(1, 26):
        image_name = getImageName(i)

        with open(path + '/' + image_name + '_predicted.txt') as f:
            lines = [line.rstrip() for line in f]

        with open(path + 'train/' + image_name + '_predicted.txt') as f:
            lines_train = [line.rstrip() for line in f]

        if lines[0] != lines_train[0]:
            print(f'Wrong dart count for file {image_name} - expected {lines_train[0]}, got {lines[0]}')

        lines, lines_train = lines[1:], lines_train[1:]
        for score in lines:
            if score in lines_train:
                accuracy += 1
            else:
                print(f'Wrong score for file {image_name} - {score} not in {lines_train}')

    print(f'Model accuracy: {accuracy / 50}')


def main(path):
    # TODO: fitting darts
    # for outliers: robust fitting, RANSAC
    # for many lines: voting methods - RANSAC, Hough transform

    task1(path)
    checkScore('evaluation/Task1')
    # task2(path)
    # task3(path)


main('train')
