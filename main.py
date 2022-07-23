from task1 import task1
from task2 import task2

from common import getImageName


def checkScore(path):
    """
    Custom eval used to check score differences

    :param path: the path where to find the train / test data
    :return: None
    """
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
    """
    Main function, it calls the solving functions for each task.

    :param path: the path where to find the train / test data
    :return: None
    """
    task1(path + '/Task1/')
    # checkScore('evaluation/Task1')
    task2(path + '/Task2/')
    # checkScore('evaluation/Task2')
    # task3(path + '/Task3/')


main('train')
