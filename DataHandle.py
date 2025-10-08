import csv
import math
import matplotlib.pyplot as plt
import numpy as np

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

    # We use Linear Interpolation method here to decide percentile. Paper: https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf
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

    def _calculate_all_stats(self):
        for column in self.column_list:
            if self.column_type[column] == 'numeric' and self.data_set[column]:
                self.stats[column] = self._calculate_column_stats(self.data_set[column])

    def load_data(self, file_path): # To Sophie 1: I'm not sure we do it or just using panda.read_csv, it's much less headache later. I use this at first because I'm afraid they said it's the library that has done all bla bla bla, to be reconsidered later...
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
        self._calculate_all_stats()
        self._display_results(self.stats)

    def plot_histogram(self):
        features = []
        for column in self.column_list:
            if self.column_type[column] == 'numeric' and self.data_set[column] and column != 'Index':
                features.append(column)
        n_rows = 3
        n_cols = (len(features) + n_rows - 1) // n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10))  
        axes = axes.flatten()          

        for idx, feature in enumerate(features):
            values = self.data_set[feature]
            sub_plot = axes[idx]  

            sub_plot.hist(values, bins = 20, edgecolor="black", alpha=0.7, color='steelblue')
            
            self._calculate_all_stats()
            std = self.stats[feature]['std']
            mean = self.stats[feature]['mean']
            cv = abs(std/mean) #coefficient of variation

            # if cv < 0.3: 
            #     assessment = "HOMOGENOUS\nx Consider REMOVING" # To Sophie 2: even CV is small, but it has several picks, we keep it too..
            #     color = 'red'
            # elif cv > 0.5: # To Sophie 3: if mean is too small, it causes CV high, but it does not say that we should keep it. If it has standard distribution, it's enough.
            #     assessment = "HETEROGENOUS\n✓ Consider KEEPING"
            #     color = 'green'
            # else:
            #     assessment = "MODERATE\n⚠️ Need more assessment"
            #     color = 'orange'
            # # To Sophie 4: in conclusion: I'm not sure it's a good idea to add the assesment or just showing the histogram and we talk on the way of how we'll choose the feature.

            sub_plot.set_title(f'{feature}\n{assessment}',fontsize=8, fontweight='bold', color=color)
            sub_plot.set_xlabel('Grade')
            sub_plot.set_ylabel('Frequency')

            sub_plot.axvline(mean, color='red', linestyle='--', linewidth=0.5)
            sub_plot.text(0.02, 0.98, f'Mean = {mean:.2f}\nStd = {std:.2f}\nCV = {cv:.2f}',
            transform=sub_plot.transAxes, fontsize=6, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            sub_plot.grid(axis='y', alpha=0.3)

        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Histogram of all subjects', fontsize = 16)
        fig.tight_layout()
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nHistograms are closed")
            plt.close('all')