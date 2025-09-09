import math

class DataDescriber:
    """Class mô phỏng pandas.describe() functionality"""
    
    def __init__(self):
        self.stats_functions = {
            'count': self._count,
            'mean': self._mean,
            'std': self._std,
            'min': self._min,
            '25%': lambda data: self._percentile(data, 0.25),
            '50%': lambda data: self._percentile(data, 0.50),
            '75%': lambda data: self._percentile(data, 0.75),
            'max': self._max
        }
    
    def describe(self, data_dict, include_all=False):
        """
        Tính toán thống kê cho dictionary data
        Args:
            data_dict: Dictionary với format {'column_name': [values]}
            include_all: Nếu True thì include cả non-numeric columns
        """
        if not data_dict:
            print("❌ Không có dữ liệu để phân tích!")
            return None
        
        # 1. Identify numeric columns
        numeric_columns = self._identify_numeric_columns(data_dict, include_all)
        
        if not numeric_columns:
            print("❌ Không tìm thấy cột numeric nào!")
            return None
        
        # 2. Calculate statistics
        results = {}
        for column in numeric_columns:
            results[column] = self._calculate_column_stats(data_dict[column])
        
        # 3. Display results
        self._display_results(results)
        
        return results
    
    def _identify_numeric_columns(self, data_dict, include_all):
        """Identify numeric columns"""
        numeric_columns = []
        
        for column_name, values in data_dict.items():
            if include_all:
                numeric_columns.append(column_name)
            else:
                # Check if column is numeric
                if self._is_numeric_column(values):
                    numeric_columns.append(column_name)
                else:
                    print(f"⏭️  Bỏ qua cột '{column_name}' (không phải numeric)")
        
        return numeric_columns
    
    def _is_numeric_column(self, values):
        """Check if a column contains numeric data"""
        if not values:
            return False
        
        numeric_count = 0
        total_count = len(values)
        
        for value in values:
            try:
                float(value)
                numeric_count += 1
            except (ValueError, TypeError):
                continue
        
        # Consider numeric if > 80% values can be converted to numbers
        return numeric_count / total_count > 0.8
    
    def _convert_to_numeric(self, values):
        """Convert values to numeric, filtering out non-numeric"""
        numeric_values = []
        
        for value in values:
            try:
                if value is not None and str(value).strip():
                    numeric_values.append(float(value))
            except (ValueError, TypeError):
                continue
        
        return numeric_values
    
    def _calculate_column_stats(self, values):
        """Calculate all statistics for a column"""
        numeric_values = self._convert_to_numeric(values)
        
        if not numeric_values:
            return {stat: None for stat in self.stats_functions.keys()}
        
        stats = {}
        for stat_name, stat_func in self.stats_functions.items():
            try:
                stats[stat_name] = stat_func(numeric_values)
            except Exception as e:
                stats[stat_name] = None
                print(f"⚠️  Lỗi tính {stat_name}: {e}")
        
        return stats
    
    # Statistical functions
    def _count(self, values):
        """Count non-null values"""
        return len(values)
    
    def _mean(self, values):
        """Calculate mean"""
        return sum(values) / len(values) if values else 0
    
    def _std(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        
        mean_val = self._mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _min(self, values):
        """Calculate minimum"""
        return min(values) if values else None
    
    def _max(self, values):
        """Calculate maximum"""
        return max(values) if values else None
    
    def _percentile(self, values, percentile):
        """Calculate percentile (0.0 to 1.0)"""
        if not values:
            return None
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if percentile == 0.0:
            return sorted_values[0]
        if percentile == 1.0:
            return sorted_values[-1]
        
        # Linear interpolation method
        index = percentile * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        weight = index - lower_index
        return (sorted_values[lower_index] * (1 - weight) + 
                sorted_values[upper_index] * weight)
    
    def _display_results(self, results):
        """Display results in a formatted table"""
        if not results:
            return
        
        print("\n" + "="*80)
        print("📊 THỐNG KÊ MÔ TẢ (DESCRIBE)")
        print("="*80)
        
        # Get column names and statistics names
        columns = list(results.keys())
        stats_names = list(self.stats_functions.keys())
        
        # Print header
        header = f"{'Statistic':<12}"
        for col in columns:
            header += f"{col:>12}"
        print(header)
        print("-" * len(header))
        
        # Print each statistic row
        for stat in stats_names:
            row = f"{stat:<12}"
            for col in columns:
                value = results[col][stat]
                if value is None:
                    row += f"{'NaN':>12}"
                elif isinstance(value, float):
                    row += f"{value:>12.2f}"
                else:
                    row += f"{value:>12}"
            print(row)
        
        print("="*80)


# Example usage và test functions
def create_sample_data():
    """Tạo dữ liệu mẫu để test"""
    return {
        'age': [25, 30, 35, 28, 32, 29, 31, 27, 33, 26],
        'salary': [50000, 60000, 75000, 55000, 65000, 58000, 70000, 52000, 68000, 56000],
        'score': [8.5, 7.2, 9.1, 6.8, 8.0, 7.5, 8.8, 7.0, 8.3, 7.7],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance']
    }

def test_describe_function():
    """Test the describe function"""
    print("🧪 TESTING CUSTOM DESCRIBE FUNCTION")
    print("="*50)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize describer
    describer = DataDescriber()
    
    # Test 1: Default behavior (only numeric columns)
    print("\n📋 Test 1: Chỉ numeric columns (default)")
    results = describer.describe(data)
    
    # Test 2: Include all columns
    print("\n📋 Test 2: Tất cả columns")
    results_all = describer.describe(data, include_all=True)
    
    return results

def compare_with_pandas():
    """So sánh với pandas describe (nếu có)"""
    try:
        import pandas as pd
        
        print("\n🔍 SO SÁNH VỚI PANDAS")
        print("="*50)
        
        data = create_sample_data()
        df = pd.DataFrame(data)
        
        print("📊 Pandas describe():")
        print(df.describe())
        
        print("\n📊 Custom describe():")
        describer = DataDescriber()
        describer.describe(data)
        
    except ImportError:
        print("\n💡 Pandas không available để so sánh")

# Advanced usage example
def advanced_example():
    """Ví dụ advanced với mixed data types"""
    print("\n🚀 ADVANCED EXAMPLE")
    print("="*50)
    
    # Data với mixed types và missing values
    mixed_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'revenue': [100.5, 200.0, None, 150.75, '250.25', 180.0, 220.5, 'N/A', 300.0, 175.5],
        'employees': ['10', '15', '8', '12', '20', '14', '18', '9', '16', '11'],
        'rating': [4.5, 3.8, 4.2, None, 4.0, 3.5, 4.8, 4.1, 3.9, 4.3],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC', 'LA', 'Miami', 'Seattle', 'Austin', 'Denver']
    }
    
    describer = DataDescriber()
    results = describer.describe(mixed_data)
    
    return results

if __name__ == "__main__":
    # Run tests
    test_describe_function()
    compare_with_pandas() 
    advanced_example()