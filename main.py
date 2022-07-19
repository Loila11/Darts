from task1 import task1
from task2 import task2


def main(path):
    # TODO: fitting darts
    # for outliers: robust fitting, RANSAC
    # for many lines: voting methods - RANSAC, Hough transform

    task1(path)
    # task2(path)
    # task3(path)


main('train')
