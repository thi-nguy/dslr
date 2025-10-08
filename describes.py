from DataHandle import DataHandle
import sys
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("How to use this program: python3 describes.py <link_file_csv>")
        print("Example: python3 describes.py data.csv")
    else:
        file_path = sys.argv[1]
        try:
            data_set = DataHandle()
            data_set.load_data(file_path)
            data_set.describe()
            # print("Compare with pandas library:\n")
            # df = pd.read_csv(file_path)
            # print(df.describe())
        except FileNotFoundError:
            print(f'File path ---{file_path}--- does not exist')
    

if __name__ == "__main__":
    main()