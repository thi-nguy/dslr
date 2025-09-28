import sys
import csv
import math

class DataHandle:
    def __init__(self):
       self.data = {}
       
       self.columnList = []
       self.columnType = {}
       self.stats = {}
       self.stats_functions = {
        'count': self._count,
        'mean': self._mean,
        'std': self._std,
        'min': self._min,
        '25%': lambda data: self._percentile(data, 25),
        '50%': lambda data: self._percentile(data, 50),
        '75%': lambda data: self._percentile(data, 75),
        'max': self._max
        }

    def loadData(self, fileName):
        try:
            with open(fileName, 'r') as file:
                reader = csv.DictReader(file)
                self.columnList = reader.fieldnames

                for column in self.columnList:
                    self.data[column] = []

                for row in reader:
                    for column in self.columnList:
                        self.data[column].append(row[column])

                for column in self.columnList:
                    isNumeric = self.isNumericalColumn(self.data[column])
                    if isNumeric:
                        self.columnType[column] = 'numeric'
                        self.data[column] = [float(val) for val in self.data[column] if val.strip()]
                    else:
                        self.columnType[column] = 'non_numeric'
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"File {fileName} does not exist")
    
    def isNumericalString(self, data):
        if isinstance(data, (int, float, str)):
            try:
                float(data)
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

    def describe(self):
        """Generate descriptive statistics for numeric columns"""
        for column in self.columnList:
            if self.columnType[column] == 'numeric' and self.data[column]:
                self.stats[column] = self._calculate_column_stats(self.data[column])
        
        self._display_results(self.stats)
    
    def _calculate_column_stats(self, column_values):
        """Calculate statistics for a single column"""
        if not column_values:
            return {}
            
        column_stats = {}
        for stat_name, stat_func in self.stats_functions.items():
            try:
                column_stats[stat_name] = stat_func(column_values)
            except Exception as e:
                print(f"Error calculating {stat_name}: {e}")
                column_stats[stat_name] = None
        
        return column_stats

    def _display_results(self, results):
        print("\n" + "=" * 80)
        print("GENERAL STATISTICS")
        print("=" * 80)

        column_names = list(results.keys())
        stat_names = list(self.stats_functions.keys())
        # print(stat_names)
        # print(columns)

        header = f"{'Statistic':<16}"
        for col in column_names:
            header += f"{col:>16}"
        print(header)
        print("-" * len(header))

        for stat in stat_names:
            print(stat)
            row = f"{stat:<10}"
            for col in column_names:
                value = results[col][stat]
                if value is None:
                    row += f"{'NaN':>16}"
                elif isinstance(value, float):
                    row += f"{value:>16.2f}"
                else:
                    row += f"{value:>16}"
            print(row)
        
        print("="*80)

    def _count(self, values):
        return len(values)

    def _mean(self, values):
        return sum(values) / len(values) if values else None

    def _min(self, values):
        return min(values) if values else None

    def _max(self, values):
        return max(values) if values else None

    def _std(self, values):
        mean_val = self._mean(values)
        variance = sum((x-mean_val) ** 2 for x in values)/ (len(values) - 1)
        return math.sqrt(variance)

    # Linear Interpolation method. Paper: https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf
    def _percentile(self, values, percentile):
        if not values:
            raise ValueError("Empty Data")
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be from 0 to 100")
        sorted_values = sorted(values)
        number_of_data = len(values)

        if percentile == 0.0:
            return sorted_values[0]
        if percentile == 1.0:
            return sorted_values[-1]
        
        index = (percentile/100) * (number_of_data - 1)
        if index == int(index):
            return sorted_values[int(index)]
        
        lower_index = int(index)
        upper_index = min(lower_index + 1, number_of_data - 1)

        if upper_index >= number_of_data:
            return sorted_data[-1]

        if (lower_index == upper_index):
            return sorted_values[lower_index]

        weight = index - lower_index
        return (sorted_values[lower_index] * (1- weight) + sorted_values[upper_index] * weight)
        
                
def main():
    dataSet = DataHandle()
    fileName = sys.argv[1]
    if not fileName:
        print(f"Need dataset in here.")
    else:
        print(fileName)
        dataSet.loadData(fileName)
        dataSet.describe()
    

if __name__ == "__main__":
    main()