⚙️ Workflow Overview

The first step in the calibration process is to generate a homography matrix. This matrix allows the system to understand the camera's perspective relative to the ground plane, enabling the conversion of pixel coordinates to real-world coordinates (X, Y).

---

## 🛠️ Field Calibration (Pixel to World Coordinates)

This crucial first step calibrates the camera's perspective, creating a transformation matrix (`homography_matrices.pkl`) that maps pixel coordinates from the camera image to real-world coordinates on the robot's workspace (the "field").

### Step 1: Prepare Input Files

You will need two files to begin:

1.  **`field_left.png`**: Your reference image. This should be a clear, top-down image of the robot's entire workspace, captured by the left camera. This image will be used to visually select calibration points.
2.  **`left_camera_dot_mapping.txt`**: Create this text file manually. It will store the coordinate pairs that link the real world to the image.

### Step 2: Populate the Coordinate Mapping File

This is the most important part of the calibration. You must identify corresponding points between the real world and your `field_left.png` image.

1.  Open `field_left.png` in an image editor that displays pixel coordinates (e.g., GIMP, Photoshop, Paint.NET).
2.  Open `left_camera_dot_mapping.txt` in a text editor.
3.  For each point you choose, add a new line to the `.txt` file in the following strict format:

    `X_map,Y_map;X_pixel,Y_pixel`

    -   **`X_map,Y_map`**: The known real-world coordinates of a point (e.g., in millimeters) within the robot's workspace.
    -   **`X_pixel,Y_pixel`**: The corresponding pixel coordinates of that exact same point in the `field_left.png` image.

    **Example `left_camera_dot_mapping.txt`:**
    ```
    120,0;902;427
    -120,0;105;425
    0,200;478;150
    0,-200;515;680
    ```

    > **Pro Tip:** For best accuracy, choose at least 4-6 points. Ensure they are spread out and cover the edges and corners of the workspace. The more points you provide, the better the calibration will be.

### Step 3: Run the Calibration Script

Once your `left_camera_dot_mapping.txt` file is populated, run the calibration script from your terminal:

```bash
python pixel_to_field_calibration_revised.py
```

After the script finishes, a new file named **`homography_matrices.pkl`** will be generated. This file contains the calculated transformation matrix, which is a required input for later stages of the project.
"""
