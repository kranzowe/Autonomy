#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time
# Match WASD node defaults
DEFAULT_SPEED = 0.38
DEFAULT_TURN_RATE = 7.0
CENTERING_THRESHOLD = 0.5  # meters before correcting
CENTER = 0.6

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

        self.wall_hit = False
        self.wall_hit_counter = 0
        self.previous_dist = None

    def get_range_at_angle(self, msg, angle_deg, window_deg=3.0):
        """Get median range at a given angle (180=front, -90=left, 90=right)."""
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
        fronts = [self.get_range_at_angle(msg, x) for x in np.arange(20.0, -20.0, -1.0)]
        valid_fronts = [r for r in fronts if np.isfinite(r)]
        avg_fronts = np.mean(valid_fronts) if valid_fronts else 12.0
        left  = self.get_range_at_angle(msg,  90.0)
        right = self.get_range_at_angle(msg, -90.0)
        rights = [self.get_range_at_angle(msg, x) for x in np.arange(-50.0, -115.0, -5.0)]
        valid_rights = [r for r in rights if np.isfinite(r)]
        avg_right = np.mean(valid_rights) if valid_rights else 12.0
        min_right = np.min(valid_rights)
        angles = np.arange(-50.0, -115.0, -5.0)
        min_angle = angles[np.argmin(rights)] 
        self.get_logger().info(f'front: {front:.2f}  left: {left:.2f}  min_angle: {min_angle:.2f} min_r: {min_right}')

        twist = Twist()

        if self.wall_hit_counter > 10:
            self.wall_hit = False
            self.wall_hit_counter = 0

        if self.wall_hit:
            twist.linear.x = -0.7*DEFAULT_SPEED
            twist.angular.z = DEFAULT_TURN_RATE

            self.wall_hit_counter += 1
        
        # elif avg_fronts < 1.0:
        #     # going straight towards a wall. Turn right
        #     twist.linear.x = -0.6*DEFAULT_SPEED
        #     twist.angular.z = DEFAULT_TURN_RATE

        elif front < 0.56 :
            self.wall_hit = True
            twist.linear.x = -0.7*DEFAULT_SPEED
            twist.angular.z = DEFAULT_TURN_RATE

        # elif front < 1.0: #and avg_right > 3.0:
        #     # Wall ahead — turn right (positive angular.z = left in ROS, so negative = right)
        #     twist.linear.x = DEFAULT_SPEED
        #     twist.angular.z = DEFAULT_TURN_RATE
        #     self.get_logger().warn(f'Wall ahead ({front:.2f}m) — turning right')

        else:
            dist_error = CENTER - min_right
            angle_error = -90 - min_angle

            if self.previous_dist is not None:
                dist_derivative = min_right - self.previous_dist
            else:
                dist_derivative = 0.0
            self.previous_dist = min_right


            twist.linear.x = DEFAULT_SPEED
            dist_component = -dist_error * 1.0
            angle_component = angle_error * 0.30 # was 0.15
            deriv_component = -dist_derivative * 0.20
            twist.angular.z = dist_component + angle_component + deriv_component
            
            self.get_logger().info(
                f'dist_err: {dist_error:.2f} → {dist_component:.2f} | '
                f'angle_err: {angle_error:.2f} → {angle_component:.2f} | '
                f'total_z: {twist.angular.z:.2f}')
            # # Drive forward
            # twist.linear.x = DEFAULT_SPEED

            # # Center between walls
            # error = left - right  # positive = too close to right, negative = too close to left
            # if error > CENTERING_THRESHOLD:
            #     twist.angular.z = -DEFAULT_TURN_RATE * 0.5   # nudge left
            #     self.get_logger().info(f'Correcting left (error: {error:.2f}m)')
            # elif error < -CENTERING_THRESHOLD:
            #     twist.angular.z = DEFAULT_TURN_RATE * 0.5  # nudge right
            #     self.get_logger().info(f'Correcting right (error: {error:.2f}m)')
            # else:
            #     twist.angular.z = 0.0
            #     self.get_logger().info('Centered — driving straight')

        self.cmd_pub.publish(twist)


def main(args=None):
    print('Waiting 15 seconds before starting...')
    time.sleep(15)
    print('Starting wall follower node!')
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
