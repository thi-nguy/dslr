from DataHandle import DataHandle
import matplotlib.pyplot as plt
import sys


def main():
    data_set = DataHandle()
    if len(sys.argv) < 2:
        print("How to use this program: python3 describes.py <link_file_csv>"
        print("Example: python3 describes.py data.csv")
    else:
        file_path = sys.argv[1] # To Sophie 5, to check the subjec, it saids a program that run directly and not taking any arg, but I don't know which dataset they want us to run so I put it like this.
        data_set.load_data(file_path)
        data_set.plot_histogram()

if __name__ == '__main__':
    main()