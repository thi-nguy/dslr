from DataHandle import DataHandle
import matplotlib.pyplot as plt
import sys


def main():
    if len(sys.argv) > 2:
        print("How to use this program: python3 describes.py <link_file_csv>")
        print("Example: python3 describes.py data.csv")
    else:
        try:
            data_set = DataHandle()
            file_path = sys.arg[1]
            data_set.load_data('dataset_train.csv')
            data_set.plot_histogram()
        except FileNotFoundError:
            print(f'File path {file_path} does not exist')

if __name__ == '__main__':
    main()