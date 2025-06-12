import pickle
import numpy as np
import cv2

class CoordinateMatcher:
    def __init__(self, pickle_path):
        # Load homography matrices from pickle file
        with open(pickle_path, "rb") as f:
            H = pickle.load(f)
        self.H_pixel2real = H.get("pixel2real")
        self.H_real2pixel = H.get("real2pixel")  # Optional, if needed in future

        if self.H_pixel2real is None:
            raise ValueError("Missing 'pixel2real' matrix in the pickle file.")

    @staticmethod
    def apply_homography(H, pts):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H)
        return mapped.reshape(-1, 2)

    def pixel_to_real(self, pixel_point, decimals=2):
        """
        Convert a pixel coordinate to a real-world coordinate.
        :param pixel_point: tuple or list of (x, y)
        :param decimals: how many decimal places to round to
        :return: tuple (x_real, y_real)
        """
        real = self.apply_homography(self.H_pixel2real, [pixel_point])[0]
        return tuple(round(val, decimals) for val in real)

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    converter = CoordinateMatcher("homography_matrices.pkl")
    pixel_coord = (878, 451)
    real_coord = converter.pixel_to_real(pixel_coord)
    print("Real coordinate (rounded):", real_coord)
