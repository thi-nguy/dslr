from DataHandle import DataHandle
import matplotlib.pyplot as plt
import sys


def main():
    if len(sys.argv) > 1:
        print("This program does not need arguments")
        print("It draws histogram for dataset_train.csv")
    else:
        try:
            data_set = DataHandle()
            data_set.load_data('dataset_train.csv')
            data_set.plot_histogram()
        except FileNotFoundError:
            print(f'File dataset_train.csv does not exist')

if __name__ == '__main__':
    main()