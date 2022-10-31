# Darts
Automatic darts scoring system

### The libraries required to run the project including the full version of each library:
- numpy==1.21.0
- cv2==4.4.0
- shapely==1.8.2

### How to run each task and where to look for the output file:

Prerequisites:
- in the code folder there must be a folder 'evaluation' with 3 subfolders: 'Task1', 'Task2' and 'Task3'

All tasks:
- script: [main.py](./main.py)
- function: main(input_folder_name), where input_folder_name is the path to the folder containing the images for all tasks, separated in folders named 'Task1', 'Task2' and 'Task3'
- output: the output file consists of txt files in [evaluation/...](./evaluation/)

Task 1: 
- script: [task1.py](./task1.py)
- function: task1(input_folder_name), where input_folder_name is the path to the folder containing the images for task1
- output: the output file consists of txt files in [evaluation/Task1/...](./evaluation/Task1/)

Task 2:
- script: [task2.py](./task2.py)
- function: task2(input_folder_name), where input_folder_name is the path to the folder containing the images for task2
- output: the output file consists of txt files in [evaluation/Task2/...](./evaluation/Task2/)

Task 3:
- script: [task3.py](./task3.py)
- function: task3(input_folder_name), where input_folder_name is the path to the folder containing the images for task3
- output: the output file consists of txt files in [evaluation/Task3/...](./evaluation/Task3/)
