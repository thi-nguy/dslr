import sys

def _displayResult():
        print("\n" + "=" * 40)
        print("GENERAL STATISTICS")
        print("=" * 40)

def main():
    data = {
    'Toán': [8.5, 7.0, 9.2, 6.8, 8.9],
    'Lý': [7.5, 8.0, 8.5, 6.0, 9.0]
    }

    for key, values in data.items():
        print(f"{key}: Max={max(values)}, Min={min(values)}, Mean={sum(values)/len(values):.2f}")
        print(f"values are: {values}")
    
    _displayResult()

if __name__ == "__main__":
    main()