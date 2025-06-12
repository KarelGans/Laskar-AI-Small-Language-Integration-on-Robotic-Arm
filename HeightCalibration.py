import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import glob
import csv
import sys

# ==============================================================================
# == 1. StereoDepthEstimator Class                                          ==
# ==============================================================================
class StereoDepthEstimator:
    """
    A class to handle object detection and depth estimation from stereo images.
    """

    def __init__(self, model_path, f_px, baseline_mm,
                 orb_features=500, bf_norm=cv2.NORM_HAMMING,
                 ratio_thresh=0.75, min_match_count=3,
                 epipolar_thresh=2.0, conf_thresh=0.3,  # Changed default conf_thresh
                 y_min=100, y_max=1000):
        """
        Initializes the StereoDepthEstimator.
        """
        print("🚀 Initializing Stereo Depth Estimator...")
        try:
            self.model = YOLO(model_path)
            print("✅ YOLO model loaded.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load YOLO model: {e}")
            raise

        self.f_px = f_px
        self.baseline_mm = baseline_mm
        self.conf_thresh = conf_thresh
        self.y_min = y_min
        self.y_max = y_max

        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.bf = cv2.BFMatcher(bf_norm, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count
        self.epipolar_thresh = epipolar_thresh

        print("✅ Estimator initialized.")

    def _detect_objects(self, image, image_path_log):
        """
        Detects objects in a single *RECTIFIED* image.
        Returns detections dictionary and the processed image for display.
        """
        processed_image = image.copy() # Work on a copy for drawing
        height, width = processed_image.shape[:2]

        # Draw shaded detection area
        overlay = processed_image.copy()
        cv2.rectangle(overlay, (0, self.y_min), (width, self.y_max), (0, 0, 255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, processed_image, 1 - alpha, 0, processed_image)

        # Draw solid horizontal lines
        cv2.line(processed_image, (0, self.y_min), (width, self.y_min), (0, 0, 255), 2)
        cv2.line(processed_image, (0, self.y_max), (width, self.y_max), (0, 0, 255), 2)

        results = self.model.predict(image, conf=self.conf_thresh, verbose=False)

        detections = {}
        for result in results:
            for box in result.boxes:
                if box.conf[0] < self.conf_thresh: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0] # Get confidence for drawing
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id] if self.model.names else str(cls_id)
                cy = int((y1 + y2) / 2)
                if not (self.y_min <= cy <= self.y_max): continue

                if label not in detections: detections[label] = []
                detections[label].append((x1, y1, x2, y2))

                # Draw on the processed_image copy
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return detections, processed_image

    def _match_and_find_points(self, left_dets, right_dets, gray_left, gray_right, 
                               left_proc_img_to_draw_on, right_proc_img_to_draw_on):
        """
        Matches objects, finds points (ORB or Centroid), AND DRAWS on provided images.
        Returns a list: [(label, centroid_L, orb_L, pt_L, pt_R), ...]
        """
        matched_data = []

        for label, l_boxes in left_dets.items():
            if label in right_dets:
                r_boxes = right_dets[label]
                num_matches = min(len(l_boxes), len(r_boxes))

                for i in range(num_matches):
                    l_x1, l_y1, l_x2, l_y2 = l_boxes[i]
                    r_x1, r_y1, r_x2, r_y2 = r_boxes[i]

                    lcx, lcy = int((l_x1 + l_x2) / 2), int((l_y1 + l_y2) / 2)
                    rcx, rcy = int((r_x1 + r_x2) / 2), int((r_y1 + r_y2) / 2)
                    centroid_left = (lcx, lcy)

                    roi_left = gray_left[l_y1:l_y2, l_x1:l_x2]
                    roi_right = gray_right[r_y1:r_y2, r_x1:r_x2]

                    good_matches = []
                    kp_left, kp_right = None, None

                    if roi_left.size > 0 and roi_right.size > 0:
                        try:
                            kp_left, des_left = self.orb.detectAndCompute(roi_left, None)
                            kp_right, des_right = self.orb.detectAndCompute(roi_right, None)

                            if des_left is not None and des_right is not None and \
                               len(kp_left) >= self.min_match_count and \
                               len(kp_right) >= self.min_match_count:
                                
                                matches = self.bf.knnMatch(des_left, des_right, k=2)
                                for m_pair in matches:
                                    if len(m_pair) == 2:
                                        m, n = m_pair
                                        if m.distance < self.ratio_thresh * n.distance:
                                            pt_l_roi = kp_left[m.queryIdx].pt
                                            pt_r_roi = kp_right[m.trainIdx].pt
                                            abs_ly = pt_l_roi[1] + l_y1
                                            abs_ry = pt_r_roi[1] + r_y1
                                            if abs(abs_ly - abs_ry) < self.epipolar_thresh:
                                                good_matches.append(m)
                        except cv2.error as e:
                            print(f"  ⚠️ OpenCV Error during ORB for {label} #{i+1}: {e}")
                            good_matches = []
                        except Exception as e:
                            print(f"  ⚠️ General Error during ORB for {label} #{i+1}: {e}")
                            good_matches = []

                    if not good_matches:
                        print(f"  ⚠️ No ORB matches for {label} #{i+1}. Falling back to centroid.")
                        pt_L, pt_R = (lcx, lcy), (rcx, rcy)
                        orb_L = None
                        cv2.circle(left_proc_img_to_draw_on, pt_L, 5, (255, 0, 0), -1) # Blue for centroid
                        cv2.circle(right_proc_img_to_draw_on, pt_R, 5, (255, 0, 0), -1) # Blue
                    else:
                        roi_center_x = (l_x2 - l_x1) / 2
                        roi_center_y = (l_y2 - l_y1) / 2
                        best_match = min(good_matches, key=lambda x: math.hypot(kp_left[x.queryIdx].pt[0] - roi_center_x, kp_left[x.queryIdx].pt[1] - roi_center_y))
                        
                        pt_l_roi = kp_left[best_match.queryIdx].pt
                        pt_r_roi = kp_right[best_match.trainIdx].pt
                        pt_L = (int(pt_l_roi[0] + l_x1), int(pt_l_roi[1] + l_y1))
                        pt_R = (int(pt_r_roi[0] + r_x1), int(pt_r_roi[1] + r_y1))
                        orb_L = pt_L
                        cv2.circle(left_proc_img_to_draw_on, pt_L, 5, (255, 0, 255), -1) # Magenta for ORB
                        cv2.circle(right_proc_img_to_draw_on, pt_R, 5, (255, 0, 255), -1) # Magenta
                        print(f"  ✅ Found ORB match for {label} #{i+1}.")

                    matched_data.append((label, centroid_left, orb_L, pt_L, pt_R))
        
        return matched_data # Processed images are modified in place

    def _compute_depth(self, pt_left, pt_right):
        """Computes depth from a single pair of points."""
        lx, _ = pt_left
        rx, _ = pt_right
        disparity = abs(lx - rx)
        if disparity == 0:
            return float('inf')
        else:
            return (self.f_px * self.baseline_mm) / disparity

    def process_pair(self, left_image_path, right_image_path):
        """
        Processes a stereo image pair for depth and returns data + processed images.
        This method is for individual pair processing with visualization.
        """
        print(f"\n--- Processing Pair: {left_image_path} & {right_image_path} ---")
        left_img_orig = cv2.imread(left_image_path)
        right_img_orig = cv2.imread(right_image_path)

        if left_img_orig is None or right_img_orig is None:
            print("❌ Failed to load one or both images.")
            return None, None, None

        left_gray = cv2.cvtColor(left_img_orig, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img_orig, cv2.COLOR_BGR2GRAY)

        # _detect_objects returns detections and the image with boxes/ROI drawn
        left_dets, left_proc_img = self._detect_objects(left_img_orig, left_image_path)
        right_dets, right_proc_img = self._detect_objects(right_img_orig, right_image_path)
        
        # _match_and_find_points will draw on left_proc_img and right_proc_img
        matched_data = self._match_and_find_points(
            left_dets, right_dets,
            left_gray, right_gray,
            left_proc_img, right_proc_img # Pass images for drawing
        )

        final_results = []
        for label, centroid_L, orb_L, pt_L, pt_R in matched_data:
            depth = self._compute_depth(pt_L, pt_R)
            final_results.append([label, centroid_L, orb_L, depth])
            print(f"🎯 {label} — Depth: {depth:.2f} mm")

        return final_results, left_proc_img, right_proc_img

    # This separate method is for CSV batch processing without needing processed images back
    def process_pair_for_csv_data(self, left_image_path, right_image_path):
        """
        Processes a pair and returns data specifically for CSV output.
        No image drawing involved directly for efficiency.
        """
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)
        if left_img is None or right_img is None: return None

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Use a temporary copy for detection drawing if needed, or avoid drawing
        # For CSV, we only need detections, not the drawn image from _detect_objects
        temp_left_img = left_img.copy() 
        temp_right_img = right_img.copy()

        left_dets, _ = self._detect_objects(temp_left_img, left_image_path) # Ignore processed image
        right_dets, _ = self._detect_objects(temp_right_img, right_image_path) # Ignore processed image

        # Pass temporary images for drawing, but these won't be returned
        matched_data = self._match_and_find_points(
            left_dets, right_dets, 
            left_gray, right_gray,
            temp_left_img, temp_right_img # ORB matches will be drawn here if needed
        )

        results_for_csv = []
        for label, centroid_L, orb_L, pt_L, pt_R in matched_data:
            depth = self._compute_depth(pt_L, pt_R)
            results_for_csv.append([label, depth]) # For CSV: obj_label, depth

        return results_for_csv


    @staticmethod
    def display_images(image_list, display_scale=0.5):
        """Static method to display a list of images."""
        for window_name, image in image_list:
            if image is not None:
                display_image = cv2.resize(image, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_image)
        print("\nPress any key to close image windows (ensure a window is active)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==============================================================================
# == 2. Batch Processing and CSV Generation                                 ==
# ==============================================================================
def process_folders_to_csv(left_folder, right_folder, output_csv, estimator, display_each_pair=False):
    """
    Iterates through image pairs in folders, processes them, and writes to CSV.
    Optionally displays each processed pair.
    """
    print(f"\n--- Starting Batch Processing ---")
    print(f"Left Folder:  {left_folder}")
    print(f"Right Folder: {right_folder}")
    print(f"Output CSV:   {output_csv}")
    print(f"Display Images: {display_each_pair}")

    left_images = sorted(glob.glob(os.path.join(left_folder, '*.jpg'))) 
    
    if not left_images:
        print(f"❌ ERROR: No images found in {left_folder}. Please check the path and extension.")
        return

    print(f"Found {len(left_images)} images in left folder.")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'object_label', 'real_height', 'calculated_depth_mm'])

        processed_count = 0
        for left_path in left_images:
            base_name = os.path.basename(left_path)
            right_path = os.path.join(right_folder, base_name)

            if not os.path.exists(right_path):
                print(f"  ⚠️ Skipping {base_name}: Corresponding right image not found.")
                continue

            try:
                parts = base_name.rsplit('.', 1)[0].split('_')
                real_height = float(parts[-2]) 
                print(f"\n▶️  Processing {base_name} (Real Height: {real_height})...")
            except (IndexError, ValueError):
                print(f"  ⚠️ Skipping {base_name}: Could not parse real height from filename.")
                continue

            if display_each_pair:
                # Use the process_pair method which returns images for display
                results, processed_left, processed_right = estimator.process_pair(left_path, right_path)
                if results:
                    processed_count += 1
                    for label, centroid_L, orb_L, depth in results: # Results from process_pair
                        if depth != float('inf'):
                            writer.writerow([base_name, label, real_height, f"{depth:.2f}"])
                            print(f"    -> Wrote: {label}, {real_height}, {depth:.2f} mm")
                        else:
                            print(f"    -> Skipped: {label} (Infinite depth)")
                    
                    # Display after writing to CSV for this pair
                    StereoDepthEstimator.display_images([
                        (f"Left: {base_name}", processed_left),
                        (f"Right: {base_name}", processed_right)
                    ])
                else:
                    print(f"   -> No results found for {base_name} when trying to display.")

            else: # No display, use the CSV-specific method
                results_csv_data = estimator.process_pair_for_csv_data(left_path, right_path)
                if results_csv_data:
                    processed_count += 1
                    for label, depth in results_csv_data:
                        if depth != float('inf'):
                            writer.writerow([base_name, label, real_height, f"{depth:.2f}"])
                            print(f"    -> Wrote: {label}, {real_height}, {depth:.2f} mm")
                        else:
                            print(f"    -> Skipped: {label} (Infinite depth)")
                else:
                    print(f"   -> No results for CSV found for {base_name}.")


    print(f"\n✅ Batch Processing finished. Processed {processed_count} pairs.")
    print(f"   Results saved to {output_csv}")


# ==============================================================================
# == 3. Main Execution Block                                                  ==
# ==============================================================================
if __name__ == "__main__":
    # --- Folders and Files ---
    LEFT_FOLDER = "calibration_height_left"
    RIGHT_FOLDER = "calibration_height_right"
    OUTPUT_CSV_FILE = "height_depth_results.csv"

    # --- YOLO Model ---
    MODEL_PATH = "best.pt"

    # --- Calibration & System Parameters ---
    CALIBRATED_F_PX = 1237.9254
    CALIBRATED_BASELINE_MM = 61.6055
    CALIBRATED_BASELINE_MM = 60 
    CONF_THRESHOLD = 0.3 # Adjusted to match class default
    Y_MIN = 100          # Adjusted to match class default
    Y_MAX = 1000         # Adjusted to match class default

    # --- NEW: Control Display ---
    DISPLAY_PROCESSED_IMAGES_DURING_BATCH = True # Set to False to skip display

    # 1. Create an instance of the estimator
    try:
        estimator = StereoDepthEstimator(
            model_path=MODEL_PATH,
            f_px=CALIBRATED_F_PX,
            baseline_mm=CALIBRATED_BASELINE_MM,
            conf_thresh=CONF_THRESHOLD,
            y_min=Y_MIN,
            y_max=Y_MAX
        )
    except Exception as e:
        print(f"❌ Failed to initialize estimator: {e}")
        sys.exit(1)

    # 2. Run the folder processing to generate the CSV
    process_folders_to_csv(
        LEFT_FOLDER, 
        RIGHT_FOLDER, 
        OUTPUT_CSV_FILE, 
        estimator,
        display_each_pair=DISPLAY_PROCESSED_IMAGES_DURING_BATCH # Pass the display flag
    )
