import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # For reshaping X

def plot_depth_to_height_regression(csv_filepath):
    """
    Reads a CSV file, converts 'real_height' from cm to mm,
    plots calculated_depth_mm vs. real_height_mm,
    and performs a linear regression to predict real_height from depth.

    Args:
        csv_filepath (str): The path to the CSV file.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"❌ Error: CSV file not found at '{csv_filepath}'")
        return

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_filepath)
        print("📄 CSV file loaded successfully. Columns found:")
        print(df.columns)
        print(f"\nFirst 5 rows of the DataFrame:\n{df.head()}")

        # --- Data Validation and Cleaning ---
        required_cols = ['real_height', 'calculated_depth_mm']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ Error: Required column '{col}' not found in the CSV file.")
                print(f"   Available columns are: {list(df.columns)}")
                return
        
        df['real_height'] = pd.to_numeric(df['real_height'], errors='coerce')
        df['real_height_mm'] = df['real_height'] * 10
        
        df['calculated_depth_mm'] = pd.to_numeric(df['calculated_depth_mm'], errors='coerce')

        df.dropna(subset=['real_height_mm', 'calculated_depth_mm'], inplace=True)

        if df.empty or len(df) < 2: # Need at least 2 points for regression
            print("❌ Error: Not enough valid data to plot or perform regression after cleaning and conversion.")
            return

        print(f"\nDataFrame after conversion and cleaning (first 5 rows):\n{df.head()}")

        # --- Linear Regression: Predicting Real Height from Calculated Depth ---
        # X is now 'calculated_depth_mm', y is 'real_height_mm'
        X = df['calculated_depth_mm'].values.reshape(-1, 1) 
        y = df['real_height_mm'].values

        # Create a linear regression model
        model = LinearRegression()

        # Fit the model
        model.fit(X, y)

        # Make predictions (predicting real_height_mm)
        y_pred = model.predict(X)

        # Get model parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print("\n--- Linear Regression Results (Predicting Height from Depth) ---")
        print(f"Slope (Coefficient): {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("Equation: real_height_mm = {:.4f} * calculated_depth_mm + {:.4f}".format(slope, intercept))


        # --- Plotting ---
        plt.figure(figsize=(12, 7))
        
        # Scatter plot of the actual data
        # X-axis is calculated_depth_mm, Y-axis is real_height_mm
        plt.scatter(X, y, alpha=0.7, edgecolors='w', s=50, label='Actual Data')
        
        # Plot the regression line
        plt.plot(X, y_pred, color='green', linewidth=2, label='Linear Regression Line (Predicting Height)')
        
        # Add labels and title
        plt.xlabel("Calculated Depth (mm)") # X-axis label changed
        plt.ylabel("Real Height (mm)")     # Y-axis label changed
        plt.title("Calculated Depth vs. Real Height with Linear Regression") # Title changed
        
        # Add a grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend()
        
        # Show the plot
        print("\n📊 Displaying plot...")
        plt.show()
        print("✅ Plot displayed.")

    except pd.errors.EmptyDataError:
        print(f"❌ Error: The CSV file '{csv_filepath}' is empty.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# ====================
#      MAIN SCRIPT
# ====================
if __name__ == "__main__":
    # Define the path to your CSV file
    CSV_FILE_PATH = "height_depth_results.csv" 
    
    # Call the function to plot the data
    plot_depth_to_height_regression(CSV_FILE_PATH)
