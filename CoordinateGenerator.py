import cv2
import numpy as np
from ultralytics import YOLO
import math
from CoordinateMatcher import CoordinateMatcher
import json

class StereoDepthEstimator:
    def __init__(self, model_path="best.pt", f_px=1237.9254, baseline_mm=60,
                 orb_features=500, bf_norm=cv2.NORM_HAMMING,
                 ratio_thresh=0.75, min_match_count=3,
                 epipolar_thresh=2.0, conf_thresh=0.5,
                 y_min=200, y_max=1000):
        
        """
        Initializes the StereoDepthEstimator.

        Args:
            model_path (str): Path to the YOLO model file.
            f_px (float): Calibrated focal length in pixels.
            baseline_mm (float): Calibrated baseline in millimeters.
            orb_features (int): Max ORB features to detect.
            bf_norm: ORB matcher norm type.
            ratio_thresh (float): Lowe's ratio test threshold.
            min_match_count (int): Minimum ORB matches needed.
            epipolar_thresh (float): Max Y-diff for ORB matches.
            conf_thresh (float): YOLO confidence threshold.
            y_min (int): Min Y for ROI.
            y_max (int): Max Y for ROI.
        """
        print("Initializing Stereo Depth Estimator...")
        try:
            self.model = YOLO(model_path)
            print("YOLO model loaded.")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}")
            raise

        self.f_px = f_px
        self.baseline_mm = baseline_mm
        self.conf_thresh = conf_thresh
        self.y_min = y_min
        self.y_max = y_max

        # ORB and Matcher Setup
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.bf = cv2.BFMatcher(bf_norm, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count
        self.epipolar_thresh = epipolar_thresh

        print("Estimator initialized.")
        
    def format_result_to_json(self, coordinate_result_array):

        # An empty dictionary to store the formatted results
        object_coordinates = {}

        # Iterate over each item in the data list
        for item in coordinate_result_array:
            # Extract the required values from each inner list
            obj_name = item[0]
            x_coord = item[1][0]
            y_coord = item[1][1]
            z_coord = item[3]

            # Assign the values to the dictionary with the object name as the key
            object_coordinates[obj_name] = {
                'x': round(float(x_coord), 2),
                'y': round(float(y_coord), 2),
                'z': round(float(z_coord), 2)
            }

        # Convert the Python dictionary to a JSON formatted string
        # indent=4 makes the output readable
        return json.dumps(object_coordinates, indent=4)


    def _detect_objects(self, image, image_path_log):
        """
        Detects objects in a single *RECTIFIED* image.
        Returns detections dictionary and the processed image.
        """
        processed_image = image.copy()
        height, width = processed_image.shape[:2]

        # Draw ROI
        overlay = processed_image.copy()
        cv2.rectangle(overlay, (0, self.y_min), (width, self.y_max), (0, 0, 255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, processed_image, 1 - alpha, 0, processed_image)
        cv2.line(processed_image, (0, self.y_min), (width, self.y_min), (0, 0, 255), 2)
        cv2.line(processed_image, (0, self.y_max), (width, self.y_max), (0, 0, 255), 2)

        # YOLO Detection
        results = self.model.predict(image, conf=self.conf_thresh, verbose=False)

        detections = {}
        for result in results:
            for box in result.boxes:
                if box.conf[0] < self.conf_thresh: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.model.names[cls_id] if self.model.names else str(cls_id)
                cy = int((y1 + y2) / 2)
                if not (self.y_min <= cy <= self.y_max): continue

                if label not in detections: detections[label] = []
                detections[label].append((x1, y1, x2, y2))

                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        print(f"Found {sum(len(v) for v in detections.values())} objects in {image_path_log} within ROI.")
        return detections, processed_image

    def _match_and_find_points(self, left_dets, right_dets, gray_left, gray_right, left_proc_img, right_proc_img):
        """
        Matches objects, finds points (ORB or Centroid), and draws them.
        Returns a list: [(label, centroid_L, orb_L, pt_L, pt_R), ...]
        orb_L is None if fallback. pt_L/pt_R are points used for depth.
        """
        matched_data = []

        for label, l_boxes in left_dets.items():
            if label in right_dets:
                r_boxes = right_dets[label]
                num_matches = min(len(l_boxes), len(r_boxes))
                print(f"Attempting to match {num_matches} instance(s) of '{label}'...")

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
                        except Exception as e:
                            print(f"Error during ORB for {label} #{i+1}: {e}")
                            good_matches = []

                    if not good_matches:
                        print(f"No ORB matches for {label} #{i+1}. Falling back to centroid.")
                        pt_L, pt_R = (lcx, lcy), (rcx, rcy)
                        orb_L = None
                        cv2.circle(left_proc_img, pt_L, 5, (255, 0, 0), -1) # Blue
                        cv2.circle(right_proc_img, pt_R, 5, (255, 0, 0), -1) # Blue
                    else:
                        roi_center_x = (l_x2 - l_x1) / 2
                        roi_center_y = (l_y2 - l_y1) / 2
                        min_dist = float('inf')
                        best_match = None
                        for match in good_matches:
                            pt_l_roi = kp_left[match.queryIdx].pt
                            dist = math.hypot(pt_l_roi[0] - roi_center_x, pt_l_roi[1] - roi_center_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_match = match
                        
                        pt_l_roi = kp_left[best_match.queryIdx].pt
                        pt_r_roi = kp_right[best_match.trainIdx].pt
                        pt_L = (int(pt_l_roi[0] + l_x1), int(pt_l_roi[1] + l_y1))
                        pt_R = (int(pt_r_roi[0] + r_x1), int(pt_r_roi[1] + r_y1))
                        orb_L = pt_L # Store the ORB point
                        cv2.circle(left_proc_img, pt_L, 5, (255, 0, 255), -1) # Magenta
                        cv2.circle(right_proc_img, pt_R, 5, (255, 0, 255), -1) # Magenta
                        print(f"Found ORB match for {label} #{i+1}.")

                    matched_data.append((label, centroid_left, orb_L, pt_L, pt_R))
        
        return matched_data

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
        Processes a stereo image pair and returns depth results.

        Args:
            left_image_path (str): Path to the left rectified image.
            right_image_path (str): Path to the right rectified image.

        Returns:
            A tuple containing:
                - list: A list of results in the format
                        [[obj_name, centroid_L, orb_L, depth_mm], ...].
                - np.ndarray: The processed left image for display.
                - np.ndarray: The processed right image for display.
            Returns (None, None, None) on image loading failure.
        """
        print(f"\n--- Processing Pair: {left_image_path} & {right_image_path} ---")
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)

        if left_img is None or right_img is None:
            print("Failed to load one or both images.")
            return None, None, None

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        left_dets, left_proc = self._detect_objects(left_img, left_image_path)
        right_dets, right_proc = self._detect_objects(right_img, right_image_path)

        print("\n--- Matching Objects & Finding Points ---")
        matched_data = self._match_and_find_points(
            left_dets, right_dets,
            left_gray, right_gray,
            left_proc, right_proc # Pass images to be drawn upon
        )

        print("\n--- Computing Depths ---")
        final_results = []
        for label, centroid_L, orb_L, pt_L, pt_R in matched_data:
            depth = self._compute_depth(pt_L, pt_R)
            final_results.append([label, centroid_L, orb_L, depth])
            print(f"*{label} — Depth: {depth:.2f} mm")

        return final_results, left_proc, right_proc

    @staticmethod
    def display_images(image_list, display_scale=0.5):
        """Static method to display a list of images."""
        for window_name, image in image_list:
            if image is not None:
                display_image = cv2.resize(image, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_image)
        print("\nPress any key to close images...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ====================
#      MAIN SCRIPT
# ====================
if __name__ == "__main__":
    # --- Configuration ---
    LEFT_IMAGE = "left_image_4.jpg"
    RIGHT_IMAGE = "right_image_4.jpg"
    converter = CoordinateMatcher("homography_matrices.pkl")
    
    # 1. Create an instance of the estimator
    try:
        estimator = StereoDepthEstimator()
    except Exception as e:
        print(f"Failed to initialize estimator: {e}")
        exit()

    # 2. Process the image pair
    results, processed_left, processed_right = estimator.process_pair(LEFT_IMAGE, RIGHT_IMAGE)

    # 3. Print the final results
    if results is not None:
        print("\n ======================================")
        print("           Final Results Array")
        print("   ======================================")
        for res in results:
            res[1]=converter.pixel_to_real(res[1]) #Change to real world coordinate
            res[3]= (-0.9529 * res[3] + 476.7135) - 86
            obj, cent, orb, depth = res
            orb_str = "None" if orb is None else f"({orb[0]}, {orb[1]})"
            print(f"  - [{obj!r}, ({cent[0]}, {cent[1]}), {orb_str}, {depth:.2f}]")
        print("   ======================================")
        print(results)
        print("\n Script finished.")
        print(estimator.format_result_to_json(results)) #This is for sending the formatted coordinate to the LLM
        
        # 4. Display the processed images
        estimator.display_images([
            (f"Left: {LEFT_IMAGE}", processed_left),
            (f"Right: {RIGHT_IMAGE}", processed_right)
        ])