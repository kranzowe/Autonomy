#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

# Match WASD node defaults
DEFAULT_SPEED = 0.4
DEFAULT_TURN_RATE = 10.0
CENTERING_THRESHOLD = 0.3  # meters before correcting

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('Wall follower node initialized...')

    def get_range_at_angle(self, msg, angle_deg, window_deg=3.0):
        """Get median range at a given angle (0=front, 90=left, -90=right)."""
        angle_rad = np.deg2rad(angle_deg)
        center_idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
        offset = int(np.deg2rad(window_deg) / msg.angle_increment)
        start_idx = max(0, center_idx - offset)
        end_idx = min(len(msg.ranges), center_idx + offset + 1)
        ranges = msg.ranges[start_idx:end_idx]
        valid = [r for r in ranges if msg.range_min < r < msg.range_max and np.isfinite(r)]
        return float(np.median(valid)) if valid else float('inf')

    def scan_callback(self, msg):
        front = self.get_range_at_angle(msg,   0.0)
        left  = self.get_range_at_angle(msg,  90.0)
        right = self.get_range_at_angle(msg, -90.0)

        self.get_logger().info(f'front: {front:.2f}  left: {left:.2f}  right: {right:.2f}')

        twist = Twist()

        if front < 1.0:
            # Wall ahead — turn right (positive angular.z = left in ROS, so negative = right)
            twist.linear.x = DEFAULT_SPEED
            twist.angular.z = -DEFAULT_TURN_RATE
            self.get_logger().warn(f'Wall ahead ({front:.2f}m) — turning right')

        else:
            # Drive forward
            twist.linear.x = DEFAULT_SPEED

            # Center between walls
            error = left - right  # positive = too close to right, negative = too close to left
            if error > CENTERING_THRESHOLD:
                twist.angular.z = DEFAULT_TURN_RATE * 0.5   # nudge left
                self.get_logger().info(f'Correcting left (error: {error:.2f}m)')
            elif error < -CENTERING_THRESHOLD:
                twist.angular.z = -DEFAULT_TURN_RATE * 0.5  # nudge right
                self.get_logger().info(f'Correcting right (error: {error:.2f}m)')
            else:
                twist.angular.z = 0.0
                self.get_logger().info('Centered — driving straight')

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
