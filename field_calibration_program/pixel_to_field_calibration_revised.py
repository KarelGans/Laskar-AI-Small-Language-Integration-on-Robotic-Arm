import numpy as np
import cv2
import pickle

# ------------------------------
# Save homography matrices to a pickle file
# ------------------------------
def save_homography(H_real2pixel, H_pixel2real, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'real2pixel': H_real2pixel, 'pixel2real': H_pixel2real}, f)
    print(f"Saved homography matrices to: {filename}")

# ------------------------------
# Load homography matrices from a pickle file
# ------------------------------
def load_homography(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['real2pixel'], data['pixel2real']

# ------------------------------
# Read calibration data from file
# ------------------------------
def read_calibration_file(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                real_part, pixel_x, pixel_y = line.split(';')
                real_x, real_y = map(float, real_part.split(','))
                pixel_x, pixel_y = float(pixel_x), float(pixel_y)
                data.append((real_x, real_y, pixel_x, pixel_y))
            except ValueError:
                print(f"Skipping invalid line: {line}")
    return data

# ------------------------------
# Compute homography matrices
# ------------------------------
def compute_homography(data):
    unique_data = []
    seen = set()
    for d in data:
        if (d[0], d[1], d[2], d[3]) not in seen:
            unique_data.append(d)
            seen.add((d[0], d[1], d[2], d[3]))

    real_pts = np.array([[x, y] for x, y, _, _ in unique_data], dtype=np.float32)
    pixel_pts = np.array([[px, py] for _, _, px, py in unique_data], dtype=np.float32)

    H_real2pixel, status1 = cv2.findHomography(real_pts, pixel_pts, cv2.RANSAC)
    H_pixel2real, status2 = cv2.findHomography(pixel_pts, real_pts, cv2.RANSAC)

    return H_real2pixel, H_pixel2real

# ------------------------------
# Apply homography on a list of points
# ------------------------------
def apply_homography(H, pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H)
    return mapped.reshape(-1, 2)

# ------------------------------
# Draw points on image with labels
# ------------------------------
def draw_points(image_path, coord_type, coord_list, H_real2pixel, H_pixel2real, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    for x, y in coord_list:
        if coord_type == "real":
            # real → pixel
            px, py = apply_homography(H_real2pixel, [(x, y)])[0]
            label = f"R({x:.0f},{y:.0f})"
        elif coord_type == "pixel":
            # pixel → real
            rx, ry = apply_homography(H_pixel2real, [(x, y)])[0]
            px, py = x, y
            label = f"P({int(px)},{int(py)})→R({rx:.0f},{ry:.0f})"
        else:
            raise ValueError("coord_type must be 'pixel' or 'real'")

        # Draw circle and label
        cv2.circle(image, (int(px), int(py)), 6, (0, 0, 255), -1)
        cv2.putText(image, label, (int(px) + 8, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved image with points to: {output_path}")
    else:
        cv2.imshow("Calibration Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    file_path = "left_camera_dot_mapping.txt"  # your txt file
    image_path = "field_left.jpg"               # your background image

    data = read_calibration_file(file_path)
    H_real2pixel, H_pixel2real = compute_homography(data)
    
    # Save the homography matrices to a pickle file
    save_homography(H_real2pixel, H_pixel2real, "homography_matrices.pkl")

    # Example: multiple real coords to pixel
    real_coords = []
    for i in range(10):
        for j in range(23):
            real_coords.append((100 + i*20, -220 + j*20))

    # Example: multiple pixel coords to real
    pixel_coords = [
        (902, 427),
        (907, 726),
        (1056, 723),
        (1447, 719),
        (910, 925),
    ]

    # Choose which one to draw:
    draw_points(image_path, "real", real_coords, H_real2pixel, H_pixel2real)
    # Or for pixel → real:
    # draw_points(image_path, "pixel", pixel_coords, H_real2pixel, H_pixel2real)
    

