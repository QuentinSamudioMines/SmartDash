import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class LoadCurve:
    def __init__(self, id, year, unit="kW"):
        self.id = id
        self.year = year
        self.unit = unit
        self.timestamp = pd.date_range(
            start=f"{year}-01-01", end=f"{year}-12-31 23:45:00", freq="15min"
        )
        self.curve = pd.Series(
            np.random.rand(len(self.timestamp)), index=self.timestamp
        )

    def plot_curve(self, title=None):
        """Plot the load curve with an optional custom title."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamp, self.curve, label=f"Load Curve (ID={self.id})")
        plt.xlabel("Timestamp")
        plt.ylabel(f"Load ({self.unit})")
        plt.title(title if title else f"Load Curve for Year {self.year}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_to_csv(self, filepath):
        """Save the load curve to a CSV file."""
        df = self.curve.to_frame(name="load")
        df.index.name = "timestamp"
        df.to_csv(filepath)
        print(f"LoadCurve saved to {filepath}")

    @staticmethod
    def load_from_csv(filepath, id, year, unit="kW"):
        """Load a load curve from a CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
        df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)

        # Create a LoadCurve instance
        load_curve = LoadCurve(id=id, year=year, unit=unit)
        load_curve.curve = df["load"]
        return load_curve


# Example usage
load_curve_example = LoadCurve(id=1, year=2023)
csv_file_path = "load_curve_example.csv"
load_curve_example.save_to_csv(csv_file_path)

# Load the load curve from CSV
loaded_curve_from_csv = LoadCurve.load_from_csv(csv_file_path, id=1, year=2023)

# Plot with a custom title
loaded_curve_from_csv.plot_curve(title="Custom Load Curve Title")
