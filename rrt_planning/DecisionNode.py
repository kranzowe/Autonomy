import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from HallwayDetector import HallwayDetector


class DecisionNode(Node):

    def __init__(self):
        super().__init__('decision_node')

        # -----------------------------
        # Tuning Parameters
        # -----------------------------
        self.GOAL_DISTANCE = 1.5          # meters - how far ahead to place goal
        self.GOAL_REACHED_THRESHOLD = 0.3 # meters - how close is "close enough"

        # -----------------------------
        # HallwayDetector
        # -----------------------------
        self.detector = HallwayDetector()

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)

        # -----------------------------
        # Publishers
        # -----------------------------
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # -----------------------------
        # State
        # -----------------------------
        self.current_pose = None
        self.current_goal = None
        self.goal_active = False

        self.get_logger().info('Decision node ready')

    # ==========================================================
    # Callbacks
    # ==========================================================

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Track current robot pose from slam_toolbox."""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg: LaserScan):
        """Main decision loop - runs on every LiDAR scan."""
        if self.current_pose is None:
            return

        # Check if we've reached the current goal
        self.check_goal_reached()

        if self.goal_active:
            return

        # Extract right side ranges
        ranges, angles = self.detector.get_right_ranges(msg)

        # Analyze for hallway or doorway
        result = self.detector.analyze(ranges, angles)

        self.get_logger().info(f'Detection result: {result}')

        if result == 'hallway':
            self.send_goal('right')
        else:
            # 'doorway' or 'none' - continue straight
            self.send_goal('straight')

    # ==========================================================
    # Goal Management
    # ==========================================================

    def send_goal(self, direction='straight'):
        """Calculate and publish a goal pose to the RRT planner."""
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = self.yaw_from_quaternion(self.current_pose.orientation)

        if direction == 'right':
            goal_yaw = yaw - np.pi / 2   # 90 degrees right
        else:
            goal_yaw = yaw               # straight ahead

        goal_x = x + self.GOAL_DISTANCE * np.cos(goal_yaw)
        goal_y = y + self.GOAL_DISTANCE * np.sin(goal_yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.orientation.z = np.sin(goal_yaw / 2)
        msg.pose.orientation.w = np.cos(goal_yaw / 2)

        self.goal_pub.publish(msg)

        self.current_goal = np.array([goal_x, goal_y])
        self.goal_active = True

        self.get_logger().info(
            f'Goal sent: ({goal_x:.2f}, {goal_y:.2f}) | direction: {direction}'
        )

    def check_goal_reached(self):
        """Reset goal_active when robot is close enough to current goal."""
        if self.current_goal is None:
            return

        x = self.current_pose.position.x
        y = self.current_pose.position.y
        dist = np.linalg.norm(np.array([x, y]) - self.current_goal)

        if dist < self.GOAL_REACHED_THRESHOLD:
            self.get_logger().info('Goal reached, looking for next opening')
            self.goal_active = False
            self.current_goal = None

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def yaw_from_quaternion(q):
        """Extract yaw angle from a quaternion."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()