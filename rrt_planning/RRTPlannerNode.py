import math
import numpy as np
import rclpy
from rclpy.node import Node


from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

from plan_on_graph import RRT, OccupancyGrid as RRTOccupancyGrid

class RRTPlannerNode(Node):

    def __init__(self):

        super().__init__('rrt_planner_node')

        self.grid = None
        self.grid_bounds = None

        #subs to
        # TODO, im not sure these really exist yet lol
        self.create_subscription(OccupancyGrid, '/occupancy_grid', self.grid_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        # TODO sub to the position?? instead of this
        self.start_pose = np.array([2.0, 2.0, 90.0])



        # pub to
        self.path_pub = self.create_publisher(Path, './waypoints', 10)

        self.get_logger().info('RRT planner is up and ready to cook')

    def grid_callback(self, msg: OccupancyGrid):
        """Store occupancy grid when received."""
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]

        grid_data = np.array(msg.data).reshape((height, width))
        binary_grid = grid_data > 50

        self.grid = RRTOccupancyGrid(origin, resolution, binary_grid)
        self.grid_bounds = [
            origin[0], origin[0] + width * resolution,
            origin[1], origin[1] + height * resolution
        ]
        self.get_logger().info(f'Received grid: {width}x{height}')
    
    def goal_callback(self, msg: PoseStamped):
        """Plan path when goal received."""
        if self.grid is None:
            self.get_logger().warn('No grid yet')
            return

        # Extract goal
        goal_yaw = self.yaw_from_quaternion(msg.pose.orientation)
        goal = np.array([msg.pose.position.x, msg.pose.position.y, math.degrees(goal_yaw)])

        self.get_logger().info(f'Planning to goal: {goal}')

        # Run RRT
        workspace = [
            np.array([self.grid_bounds[0], self.grid_bounds[2], -180]),
            np.array([self.grid_bounds[1], self.grid_bounds[3], 180])
        ]

        rrt = RRT(
            start=self.start_pose,
            goal=goal,
            rand_area=workspace,
            occupancy_grid=self.grid,
            expand_dis=0.3, # TODO tweak these
            max_iter=10000) # TODO probs too high?
            
        path = rrt.planning(animation=False)

        if path:
            self.publish_path(path)
        else:
            self.get_logger().error('No path found')

    def publish_path(self, path):
        """Publish path as nav_msgs/Path."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for point in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info(f'Published path with {len(path)} waypoints')

    @staticmethod
    def yaw_from_quaternion(q):
        """Extract yaw from quaternion."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)
    
def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


    # ai on how to run

    # Terminal 1: Run the node
# ros2 run rrt_planning rrt_planner_node

# # Terminal 2: Send a goal
# ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 8.0, y: 8.0}, orientation: {w: 1.0}}}"