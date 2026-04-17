import numpy as np


class HallwayDetector:

    def __init__(self):

        # -----------------------------
        # Tuning Parameters
        # -----------------------------
        self.HALLWAY_DEPTH_THRESHOLD = 1.5      # meters - minimum depth to count as open
        self.HALLWAY_MIN_DEPTH = 2.0             # meters - minimum depth to classify as hallway
        self.OPENING_MIN_WIDTH = np.deg2rad(30)  # radians - minimum width to count as opening
        self.HALLWAY_MIN_WIDTH = np.deg2rad(50)  # radians - minimum width to classify as hallway

        self.MAX_VALID_RANGE = 5.0               # meters - clip far readings (windows etc.)
        self.MIN_SEGMENT_POINTS = 5              # minimum consecutive rays to count as opening
        self.EDGE_CHECK_POINTS = 3               # rays sampled at edges to check for walls
        self.VARIANCE_THRESHOLD = 0.1            # reject flat/glass readings below this std dev

    # ==========================================================
    # Public Interface
    # ==========================================================

    def analyze(self, ranges, angles):
        """
        Full analysis pipeline. Takes raw right-side ranges and angles,
        returns a classification string.

        Args:
            ranges: np.array of range values (meters) for the right side
            angles: np.array of corresponding angle values (radians)

        Returns:
            'hallway'  - robot should turn right
            'doorway'  - robot should continue straight
            'none'     - no opening detected, continue straight
        """
        opening_found, depth, width = self.detect_opening(ranges, angles)

        if not opening_found:
            return 'none'

        return self.classify_opening(depth, width)

    def get_right_ranges(self, msg):
        """
        Extract and clean ranges on the right side of the robot (45 to 135 degrees).
        Assumes LiDAR motor faces front of robot.

        Args:
            msg: sensor_msgs/LaserScan message

        Returns:
            ranges: np.array of cleaned range values
            angles: np.array of corresponding angle values
        """
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Replace inf/nan with 0
        ranges = np.where(np.isfinite(ranges), ranges, 0.0)

        # Clip readings beyond MAX_VALID_RANGE (handles windows)
        ranges = np.where(ranges > self.MAX_VALID_RANGE, 0.0, ranges)

        # Extract right side only (45 to 135 degrees)
        mask = (angles >= np.deg2rad(45)) & (angles <= np.deg2rad(135))

        return ranges[mask], angles[mask]

    # ==========================================================
    # Detection
    # ==========================================================

    def detect_opening(self, ranges, angles):
        """
        Detect if there is a valid opening in the given ranges.

        Args:
            ranges: np.array of range values
            angles: np.array of corresponding angle values

        Returns:
            opening_found (bool)
            depth (float): average depth of the best opening in meters
            width (float): angular width of the best opening in radians
        """
        # Find rays that see far enough to be considered open
        is_open = ranges > self.HALLWAY_DEPTH_THRESHOLD

        # -----------------------------
        # Find contiguous segments
        # -----------------------------
        segments = []
        current = []

        for i, val in enumerate(is_open):
            if val:
                current.append(i)
            else:
                if len(current) >= self.MIN_SEGMENT_POINTS:
                    segments.append(current)
                current = []

        # Catch segment that runs to the end of the array
        if len(current) >= self.MIN_SEGMENT_POINTS:
            segments.append(current)

        if not segments:
            return False, 0.0, 0.0

        # Pick the widest contiguous segment
        best = max(segments, key=len)

        open_ranges = ranges[best]
        open_angles = angles[best]

        depth = np.mean(open_ranges)
        width = open_angles[-1] - open_angles[0]

        # -----------------------------
        # Reject too narrow
        # -----------------------------
        if width < self.OPENING_MIN_WIDTH:
            return False, depth, width

        # -----------------------------
        # Reject flat/glass (low variance)
        # -----------------------------
        if np.std(open_ranges) < self.VARIANCE_THRESHOLD:
            return False, depth, width

        # -----------------------------
        # Check for side walls
        # -----------------------------
        edge_n = min(self.EDGE_CHECK_POINTS, len(open_ranges) // 2)

        left_edge = open_ranges[:edge_n]
        right_edge = open_ranges[-edge_n:]

        left_wall = np.mean(left_edge) < self.HALLWAY_DEPTH_THRESHOLD
        right_wall = np.mean(right_edge) < self.HALLWAY_DEPTH_THRESHOLD

        if not (left_wall or right_wall):
            return False, depth, width

        return True, depth, width

    # ==========================================================
    # Classification
    # ==========================================================

    def classify_opening(self, depth, width):
        """
        Classify a detected opening as a hallway or doorway.

        Args:
            depth (float): average depth of opening in meters
            width (float): angular width of opening in radians

        Returns:
            'hallway' or 'doorway'
        """
        if depth > self.HALLWAY_MIN_DEPTH and width > self.HALLWAY_MIN_WIDTH:
            return 'hallway'
        return 'doorway'