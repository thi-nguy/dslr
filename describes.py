import sys
import csv

class DataHandle:
    def __init__(self):
       self.data = {}
       self.columnList = []

    def loadData(self, fileName):
        try:
            with open(fileName, 'r') as file:
                reader = csv.DictReader(file)
                self.columnList = reader.fieldnames
                self.columnType = {}

                for column in self.columnList:
                    self.data[column] = []

                for row in reader:
                    for column in self.columnList:
                        self.data[column].append(row[column])

                for column in self.columnList:
                    isNumeric = self.isNumericalColumn(self.data[column])
                    self.columnType[column] = 'numeric' if is_number else 'non_numeric'
        except FileNotFoundError:
            raise FileNotFoundError(f"File {fileName} does not exist")
    
    def isNumericalString(self, stringData):
        if isinstance(stringData, (int, float, str)):
            try:
                float(stringData)
                return True
            except ValueError:
                return False
        else:
            return False

    def isNumericalColumn(self, columnData):
        columnLength = len(columnData)
        numCount = 0
        for data in columnData:
            if self.isNumericalString(data):
                numCount += 1
        if numCount >= (70/100) * columnLength:
            return True
        else:
            return False

    def _displayResult():
        print("\n" + "=" * 40)
        print("GENERAL STATISTICS")
        print("=" * 40)

        

            
                
def main():
    dataSet = DataHandle()
    fileName = sys.argv[1]
    if not fileName:
        print(f"Need dataset in here.")
    else:
        print(fileName)
        dataSet.loadData(fileName)
    

if __name__ == "__main__":
    main()