# Depth Estimation & Calibration for IMX219-83 Stereo Camera

This guide details a method for calculating the real-world height of an object from stereo images by calibrating depth perception. The process uses point disparity from an IMX219-83 stereo camera setup and culminates in generating the 3D coordinates (X, Y, Z) of detected objects.

The project is divided into several key Python scripts:

-   **`HeightCalibration.py`**: Processes stereo test images to measure object heights and corresponding depth values, outputting them to a CSV file.
-   **`Regression.py`**: Performs a linear regression on the CSV data to find the mathematical relationship between calculated depth and actual height.
-   **`CoordinateGenerator.py`**: The main script that uses a trained model to detect objects and applies the calibration equation to calculate their real-world X, Y, and Z coordinates.
-   **`CoordinateMatcher.py`**: A helper library used to translate pixel coordinates into a robot arm's coordinate system.

---

## ⚙️ 1. Prerequisites

Before you begin, ensure the following setup is complete.

### A. Homography Matrix
You must have a pre-calibrated homography matrix file. Obtain the **`homography_matrices.pkl`** file from your separate field calibration process and place it in the root directory of this project.

### B. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### C. Prepare Calibration Images
The accuracy of the system depends on a good calibration dataset.

1.  Create two new folders: **`calibration_height_left`** and **`calibration_height_right`**.
2.  Populate these folders with stereo image pairs.
3.  Name the images using the following strict format:
    **`<label_name>_<height_cm>_<image_number>.jpg`**

    **Example:**
    - `battery_1.73_1.jpg`
    - `bottle_15.5_1.jpg`
    - `battery_1.73_2.jpg`

    Here, `height_cm` is the **actual, measured height** of the object in centimeters. It is crucial to collect images of various objects at different locations on your map to ensure a robust calibration.

---

## 🛠️ 2. Height Calibration Workflow

Follow these steps to generate the calibration equation that converts depth to height.

### Step 1: Generate Height-Depth Data
Run the `HeightCalibration.py` script. This will process your calibration images and create a **`height_depth_results.csv`** file containing the raw data needed for the next step.

```bash
python HeightCalibration.py
```

### Step 2: Perform Regression Analysis
Run the `Regression.py` script. It will analyze `height_depth_results.csv` and output the linear equation that models the relationship between the camera's calculated depth and the object's real-world height.

```bash
python Regression.py
```

The output will be an equation similar to this:

```
Equation: real_height_mm = -0.9529 * calculated_depth_mm + 476.7135
```

---

## 🎯 3. Final Coordinate Generation

Apply the calibration equation to the main script to get the final 3D coordinates of objects.

### Step 1: Update the Calibration Equation
1.  Open the **`CoordinateGenerator.py`** file.
2.  Locate the following line of code:
    ```python
    res[3]= (-0.9529 * res[3] + 476.7135) - 86
    ```
3.  Replace the expression `(-0.9529 * res[3] + 476.7135)` with the precise equation you derived from the regression step.
    - **Note:** The `- 86` is a user-defined Z-axis offset to match a specific robot's coordinate system. You may need to adjust or remove this value for your setup.

### Step 2: Configure Parameters
In `CoordinateGenerator.py`, you can also adjust other parameters inside the `__init__` method as needed:

```python
def __init__(self, model_path="best.pt", f_px=1237.9254, baseline_mm=60,
             orb_features=500, bf_norm=cv2.NORM_HAMMING,
             ratio_thresh=0.75, min_match_count=3,
             epipolar_thresh=2.0, conf_thresh=0.5,
             y_min=200, y_max=1000):
```
-   **`model_path`**: Path to your trained object detection model (e.g., `"yolov8n.pt"`). Get the model from the folder Yolov8_training
-   **`f_px`**: The camera's focal length in pixels.
-   **`baseline_mm`**: The distance between the two stereo camera lenses in millimeters.

### Step 3: Run the Program
Execute the script to start detecting objects and calculating their X, Y, and Z coordinates. The script will show two output images, showing the detected objects with their calculated coordinates.

```bash
python CoordinateGenerator.py
```