from task1 import task1
from task2 import task2
from task3 import task3

from common import getFileName


def checkScore(path, taskNot3=True):
    """
    Custom eval used to check score differences

    :param path: the path where to find the train / test data
    :param taskNot3: if the task being checked at the moment is task 3 or not. Task 3 predicted train files have a
    different name and don't have the number of thrown darts (it's always 1). Default is True
    :return: None
    """
    accuracy = 0

    for i in range(1, 26):
        file_name = getFileName(i)

        with open(path + '/' + file_name + '_predicted.txt') as f:
            lines = [line.rstrip() for line in f]

        out_name = path + 'train/' + file_name
        if taskNot3:
            out_name += '_predicted'
        out_name += '.txt'

        with open(out_name) as f:
            lines_train = [line.rstrip() for line in f]

        if taskNot3:
            if lines[0] != lines_train[0]:
                print(f'Wrong dart count for file {file_name} - expected {lines_train[0]}, got {lines[0]}')

            lines, lines_train = lines[1:], lines_train[1:]

        for score in lines:
            if score in lines_train:
                accuracy += 1
            else:
                if not taskNot3 and score[1:] == lines_train[0][1:]:
                    accuracy += 0.5
                print(f'Wrong score for file {file_name} - {score} not in {lines_train}')

    accuracy /= 25
    if taskNot3:
        accuracy /= 2
    print(f'Model accuracy: {accuracy}')


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
    task3(path + '/Task3/')
    # checkScore('evaluation/Task3', False)


main('test')
