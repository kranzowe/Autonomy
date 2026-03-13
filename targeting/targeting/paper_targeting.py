import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist
from vision_msgs.msg import BoundingBox2D


class PaperTargetingNode(Node):

    def __init__(self):
        super().__init__('targeting')
        self.subscription = self.create_subscription(
            BoundingBox2D,
            'blob_rectangle',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'control_target', 10)
        self.subscription

    def listener_callback(self, msg):
        obj_width_px = max(msg.size_x, msg.size_y)
        obj_centroid_px = np.array([msg.center.position.x, msg.center.position.y])

        FOV_px = np.array([1920, 1080])
        FOV_angle = 69

        # Assuming standard sized printer paper, unit m
        m_from_px = 0.279 / obj_width_px
        FOV_x_cm = FOV_px[0] * m_from_px
        # Dist from camera to FOV plane in m
        r = (FOV_x_cm / 2) * np.tan((180-FOV_angle)/2)
        # Distance of paper from FOV center in m
        x = (obj_centroid_px[0] - FOV_px[0]/2) * m_from_px
        # Paper angle from camera
        theta = np.atan2(x,r)
        # Paper distance d from camera
        d = np.sqrt(x**2 + r**2)

        target = Twist()
        target.linear.x = d
        target.angular.z = theta
        self.publisher.publish(target)

def main(args=None):
    rclpy.init(args=args)

    targeter = PaperTargetingNode()

    rclpy.spin(targeter)

    targeter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
