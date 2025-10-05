import sys
import csv
import math

class DataHandle:
    def __init__(self):
       self.data_set = {}
       self.column_list = []
       self.column_type = {}
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

    def _is_numerical_string(self, data):
        if isinstance(data, (int, float, str)):
            try:
                float(data)
                return True
            except ValueError:
                return False
        else:
            return False

    def _is_numerical_column(self, column_data):
        column_length = len(column_data)
        numeric_data_count = 0
        for data in column_data:
            if self._is_numerical_string(data):
                numeric_data_count += 1
        if numeric_data_count >= (70/100) * column_length:
            return True
        else:
            return False

    def _calculate_column_stats(self, column_values):
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

        header = ' ' * 10
        for col in column_names:
            header += f"{' ' * 3 + col}"
        print(header)

        for stat in stat_names:
            row = f"{stat:<10}"
            for col in column_names:
                value = results[col][stat]
                if value is None:
                    row += f"{'NaN':>{len(col) + 3}}"
                elif isinstance(value, float):
                    row += f"{value:>{len(col) + 3}.2f}"
                else:
                    row += f"{value:>{len(col) + 3}}"
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
        data_length = len(values)

        if percentile == 0.0:
            return sorted_values[0]
        if percentile == 1.0:
            return sorted_values[-1]
        
        index = (percentile/100) * (data_length - 1)
        if index == int(index):
            return sorted_values[int(index)]
        
        lower_index = int(index)
        upper_index = min(lower_index + 1, data_length - 1)

        if upper_index >= data_length:
            return sorted_data[-1]

        if (lower_index == upper_index):
            return sorted_values[lower_index]

        weight = index - lower_index
        return (sorted_values[lower_index] * (1- weight) + sorted_values[upper_index] * weight)

    def load_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                file_reader = csv.DictReader(file)
                self.column_list = file_reader.fieldnames

                for column in self.column_list:
                    self.data_set[column] = []

                for row in file_reader:
                    for column in self.column_list:
                        self.data_set[column].append(row[column])

                for column in self.column_list:
                    is_numeric = self._is_numerical_column(self.data_set[column])
                    if is_numeric:
                        self.column_type[column] = 'numeric'
                        self.data_set[column] = [float(val) for val in self.data_set[column] if val.strip()]
                    else:
                        self.column_type[column] = 'non_numeric'
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} does not exist")    

    def describe(self):
        """Generate descriptive statistics for numeric columns"""
        for column in self.column_list:
            if self.column_type[column] == 'numeric' and self.data_set[column]:
                self.stats[column] = self._calculate_column_stats(self.data_set[column])
        
        self._display_results(self.stats)
   
        
                
def main():
    data_set = DataHandle()
    if len(sys.argv) < 2:
        print("How to use this program: python3 describes.py <link_file_csv>")
        print("Example: python3 describes.py data.csv")
    else:
        file_path = sys.argv[1]
        print(file_path)
        data_set.load_data(file_path)
        data_set.describe()
    

if __name__ == "__main__":
    main()