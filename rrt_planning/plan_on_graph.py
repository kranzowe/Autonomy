"""
Rapidly Exploring Random Tree (RRT) Path Planning Implementation

This module implements the RRT algorithm for path planning, which works by:
1. Incrementally building a tree by sampling random points
2. Connecting new points to the nearest node in the tree
3. Checking for collisions and maintaining feasible paths

ROBO 5000 HW2: Path Planning

Instructions:
------------
1. Add your name below
2. Complete all sections marked with "Student Task:" comments/docstrings
3. The main algorithm steps are in the `planning` and `steer` methods
4. Test your implementation with different obstacle configurations

Full Name: Owen Kranz
"""

### GRADER## 
# howdy, if you are wondering what is up with node.path,
# i made it a part property to make it cleaner imo,
# if just creates a discrete ray from current to the parent

import math
import random
from typing import List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from copy import deepcopy
from PIL import Image
from scipy.ndimage import binary_dilation
import cv2
show_animation = False


class OccupancyGrid():

    def __init__(self, origin, resolution, image):

        self.robot_radius = 0.6

        self.origin = origin
        self.resolution = resolution
        self.image = self.inflate(image)

    def inflate(self, grid: np.ndarray) -> np.ndarray:
        """
        Inflate obstacles by robot radius using binary dilation.
        
        Args:
            grid: Binary occupancy grid (True = occupied)
            radius: Inflation radius in meters
            
        Returns:
            Inflated binary grid
        """
        # Convert radius to grid cells
        radius_cells = int(np.ceil(self.robot_radius / self.resolution))
        
        # Create circular structuring element
        size = 2 * radius_cells + 1
        y, x = np.ogrid[:size, :size]
        center = radius_cells
        mask = (x - center)**2 + (y - center)**2 <= radius_cells**2
        
        # Dilate obstacles
        inflated = binary_dilation(grid, structure=mask)
        
        return inflated

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return wx, wy



class RRT:
    """RRT path planning implementation."""

    class Node:
        """A node in the RRT tree."""

        def __init__(self, x: float, y: float, yaw: float = 0.0):
            """
            Initialize a node.

            Args:
                x: X coordinate
                y: Y coordinate
                psi: yaw angle degrees
            """
            self.x = x
            self.y = y
            self.psi = yaw

            self.parent: Optional["RRT.Node"] = None

        # ADDED CODEEEEE
        # these part properties are gonna make life so much easier later on
        @property
        def path(self):

            path_points = []

            delta_h = 0.1 # step size to search along ray
            for h in np.arange(0.0, 1.0, delta_h): # should really do a subdividing alg, oh well
                x = (1-h)*self.x + h*self.parent.x # standard ray equation
                y = (1-h)*self.y + h*self.parent.y # learning this in alg motion planning
                path_points.append(np.array([x,y]))


            path_points.append(np.array([self.parent.x, self.parent.y])) # add the parent to complete the path

            return path_points
        
        @property
        def path_x(self):
            path_points = self.path
            return [pnt[0] for pnt in path_points]
        
        @property
        def path_y(self):
            path_points = self.path
            return [pnt[1] for pnt in path_points]
        
        @property
        def path_psi(self):
            path_points = self.path
            return [pnt[2] for pnt in path_points]




    class AreaBounds:
        """Bounds for the planning area."""

        def __init__(self, area: List[float]):
            """
            Initialize area bounds.

            Args:
                area: List of [xmin, xmax, ymin, ymax, psimin, psimax]
            """
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
            self.psimin = float(area[4])
            self.psimax = float(area[5])

    def __init__(
        self,
        start: List[float],
        goal: List[float],
        occupancy_grid, # idk the format 
        rand_area: List[float],
        expand_dis: float = 0.1,
        ## NOTE path resolution is not used. Its a better property on the NODE
        # see my implementation of the path part properties on the None
        # path_resolution: float = 0.5,
        goal_sample_rate: int = 5,
        max_iter: int = 500,
        play_area: Optional[List[float]] = None,
        max_steer: float = 15.0,
        box = None
    ):
        """
        Initialize the RRT planner.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacle_list: List of obstacles [(x, y, radius), ...]
            rand_area: Random sampling area [min, max]
            expand_dis: Maximum distance to expand tree
            path_resolution: Resolution for path checking
            goal_sample_rate: Rate at which goal is sampled instead of random point
            max_iter: Maximum number of iterations
            play_area: Optional bounds for valid area [xmin, xmax, ymin, ymax]
            max_steer: Maximum steering angle (radians)

        Class Attributes:
            self.start: Node object representing start position
            self.end: Node object representing goal position
            self.min_rand: Minimum value for random sampling range
            self.max_rand: Maximum value for random sampling range
            self.play_area: AreaBounds object defining valid planning space (None if unbounded)
            self.expand_dis: Maximum distance to extend tree in each iteration
            self.path_resolution: Distance between intermediate points when extending tree
            self.goal_sample_rate: Probability (in %) of sampling goal instead of random point
            self.max_iter: Maximum number of iterations for tree expansion
            self.obstacle_list: List of obstacles, each defined by (x, y, radius)
            self.node_list: List of all nodes in the tree
            self.max_steer: Maximum steering angle (radians)
        """
        # Create Node objects for start and goal positions
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        # Random sampling bounds
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        # Create AreaBounds object if play_area is specified
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        # Tree expansion parameters
        self.expand_dis = expand_dis
        # self.path_resolution = path_resolution 
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        # Environment and planning data
        self.occupancy_grid = occupancy_grid
        self.node_list = []
        self.theta_weight = 0.02
        self.max_steer = max_steer
        self.goal_accept_dist = expand_dis * 3
        self.box = box

    # ADDED CODE! A lovely part property for the final path
    @property
    def final_path(self):
        total_nodes = []
        node = self.node_list[-1]  # end node
        while node is not None:
            total_nodes.append(node)
            node = node.parent
        return total_nodes

    def get_random_node(self, box=None, center=None, angle_sector=None, max_box_dist=100) -> "RRT.Node":
        """
        Sample a random node in the state space, optionally constrained to be near a box and within an angle sector.
        """
        is_goal = False
        while True:
            random_num = random.random()
            if random_num < self.goal_sample_rate/100:
                random_node = deepcopy(self.end)
                random_node.psi = random.uniform(self.min_rand[2], self.max_rand[2])
                is_goal = True
            else:
                rand_x = random.uniform(self.min_rand[0], self.max_rand[0])
                rand_y = random.uniform(self.min_rand[1], self.max_rand[1])
                rand_psi = random.uniform(self.min_rand[2], self.max_rand[2])
                random_node = self.Node(rand_x, rand_y, rand_psi)

            # If no constraints, accept
            # if box is None or center is None or angle_sector is None:
            #     break

            # Convert world to grid for pixel units
            gx, gy = self.occupancy_grid.world_to_grid(random_node.x, random_node.y)
            pt = np.array([gx, gy])

            # Check distance to box
            if min_dist_to_box(pt, self.box) > 10:
                continue

            # # Check angle sector
            # angle = angle_from_vertical(pt, center)
            # if not angle_between(angle_sector[0], angle_sector[1], angle):
            #     continue

            break

        return random_node, is_goal
    
    def calc_dist_to_goal(self, x, y):
        """Calculate distance from current position to the goal."""
        dx = x - self.end.x
        dy = y - self.end.y

        # TODO
        return math.hypot(dx, dy)

    def generate_final_course(self, goal_ind):
        """Generate the final path from the goal node to the start node.

        Args:
            goal_ind: Index of the goal node in node_list

        Returns:
            path: List of points along the path
        """
        path = []
        node = self.node_list[goal_ind]

        # For standard RRT, just use node positions
        while node.parent is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.append((node.x, node.y))

        return path[::-1]  # Reverse path to get start-to-goal order

    def planning(
        self, animation: bool = False
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using RRT.

        Args:
            animation: Whether to show animation of tree growth

        Returns:
            path: List of points [(x1,y1), (x2,y2),...] or None if no path found

        Student Task:
        ------------
        1. Initialize the tree with the start node (add to node_list)
        2. For max_iter iterations:
           a. Sample random point using get_random_node()
           b. Find nearest node in tree
           c. Attempt to steer towards sampled point
           d. If valid new node:
              - Add to tree
              - Check if goal is reachable
              - If goal reached, extract and return path
        3. Return None if no path found
        """
        self.node_list = [self.start]

        rnd_node = None
        for i in range(self.max_iter):
            # YOUR CODE GOES HERE
            # Use get_random_node() to sample a random configuration
            rnd_node, is_goal = self.get_random_node()

            # Find nearest node in the tree
            nearest_node = self.find_nearest_node(rnd_node)

            # Steer towards the sampled node 
            proposed_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Check for collisions and validity
            if proposed_node is not None:
                # I have none returned if collision sooo that collision check is doneee
                self.node_list.append(proposed_node)

            # DO NOT ALTER THE NEXT 2 LINES
            if animation and i % 200 == 0:
                self.draw_graph(rnd_node)
            # YOUR CODE GOES HERE
            # Check if goal is reachable from new node

            if proposed_node is not None:
                vec_to_goal = np.array([self.end.x - proposed_node.x, self.end.y - proposed_node.y]) 
                distance_to_goal = np.linalg.norm(vec_to_goal)
                
                if distance_to_goal < self.goal_accept_dist:
                    
                    # still gotta check for collsion
                    self.end.parent = proposed_node
                    if self.check_collision(self.end):
                        self.node_list.append(self.end)
                        break # woohoo!
                    else:
                        self.end.parent = None # reset what doth been set

        # from current node connect to goal node and find the path - return path after plotting path (below)
        # gotta compute the final path
        if self.node_list[-1] is not self.end:
            return None
        
        final_path = self.final_path

        if True:#animation:
            # For standard RRT, plot straight lines
            path_x = [pnt.x for pnt in final_path]
            path_y = [pnt.y for pnt in final_path]
            plt.plot(
                path_x,
                path_y,
                "-r",
                linewidth=2,
                label="Final Path",
            )
        plt.legend()
        plt.pause(0.01)

        return final_path

    def steer(
        self,
        from_node: "RRT.Node",
        to_node: "RRT.Node",
        extend_length: float,
    ) -> "RRT.Node":
        """
        Steer from one node towards another within constraints.

        Args:
            from_node: Node to steer from
            to_node: Node to steer towards
            extend_length: Maximum distance to extend

        Returns:
            new_node: New node after steering

        Student Task:
        ------------
        1. Create new node at from_node location
        2. Calculate distance and angle to to_node
        3. If distance > extend_length:
           - Scale movement to respect extend_length
        4. Move towards to_node:
           - Update position incrementally using path_resolution
           - Store path points in new_node.path_x and new_node.path_y
        5. Set parent relationship
        6. Return new node
        """

        distances = np.zeros((5,))
        possible_nodes = []

        for i, turn in enumerate(np.linspace(-self.max_steer, self.max_steer, 5)):
            heading = from_node.psi + turn

            possible_node = self.Node(from_node.x + self.expand_dis*np.cos(np.deg2rad(heading)), from_node.y + self.expand_dis*np.sin(np.deg2rad(heading)), heading)
            possible_nodes.append(possible_node)

            distance = self.calc_weighted_distance(
                possible_node, to_node
            )  # This function returns the distance d from new_node to to_node and the theta represents the angle the line joining them makes with the x axis

            distances[i] = distance

        new_node = possible_nodes[np.argmin(distances)]
     
        # if distance < extend_length and theta < self.max_steer:
        #     # if to node close enough, thats our new node!
        #     new_node = deepcopy(to_node)
        # else:
        #     # if too far, move extension distance along unit vector
        #     ratio = self.max_steer/distance
        #     new_x = from_node.x + ratio * (to_node.x - from_node.x)
        #     new_y = from_node.y + ratio * (to_node.y - from_node.y)
        #     new_node = self.Node(new_x, new_y)

        # new node needs a parent for path
        new_node.parent = from_node
    

        collision_free = self.check_collision(new_node)

        if collision_free:
            return new_node
        
        return None # pretty sure standard RRT just throws out nodes that have collision
        
        # dont need this stuff
        # d, _ = self.calc_distance_and_angle(
        #     new_node, to_node
        # )  # We want to check if the robot has reached to_node
        # # YOUR CODE GOES HERE
        # # Check if the robot has reached to_node. If yes, add to_node coordinates to the path of new_node
        # # Add from_node as parent of new_node

        # return new_node

    @staticmethod
    def calc_distance_and_angle(
        from_node: "RRT.Node", to_node: "RRT.Node"
    ) -> Tuple[float, float]:
        """
        Calculate distance and angle between nodes.

        Args:
            from_node: Starting node
            to_node: Target node

        Returns:
            distance: Euclidean distance between nodes
            theta: Angle in radians from from_node to to_node

        Student Task:
        ------------
        1. Calculate Euclidean distance between nodes
        2. Calculate angle of line from from_node to to_node
           relative to x-axis (use math.atan2)
        3. Return distance and angle
        """
        distance = None
        theta = None
        # YOUR CODE GOES HERE
        # Write code to find the distance between from_node and to_node
        # and the angle made by the line joining from_node and to_node with the x_axis

        vector = np.array([to_node.x - from_node.x, to_node.y - from_node.y])
        distance = np.linalg.norm(vector) # l2 norm
        theta = np.arctan2(vector[1], vector[0]) - np.deg2rad(from_node.psi)  # same as math.atan2 i am just more familiar
        # also, I never use theta lol so this is pointless
        return distance, np.rad2deg(theta)
    
    def calc_weighted_distance(self,
        from_node: "RRT.Node", to_node: "RRT.Node"
    ) -> Tuple[float, float]:
        """
        Calculate distance and angle between nodes.

        Args:
            from_node: Starting node
            to_node: Target node

        Returns:
            distance: Euclidean distance between nodes
            theta: Angle in radians from from_node to to_node

        Student Task:
        ------------
        1. Calculate Euclidean distance between nodes
        2. Calculate angle of line from from_node to to_node
           relative to x-axis (use math.atan2)
        3. Return distance and angle
        """
        distance = None
        theta = None
        # YOUR CODE GOES HERE
        # Write code to find the distance between from_node and to_node
        # and the angle made by the line joining from_node and to_node with the x_axis

        vector = np.array([to_node.x - from_node.x, to_node.y - from_node.y])
        distance = np.linalg.norm(vector) # l2 norm
        diff = abs(to_node.psi - from_node.psi) % 360
        theta = min(diff, 360 - diff) # same as math.atan2 i am just more familiar
        # also, I never use theta lol so this is pointless
        return distance + self.theta_weight*theta

    def find_nearest_node(self, node: "RRT.Node") -> "RRT.Node":
        """
        Find the nearest node in the tree to the given node.

        Args:
            node: Node to find nearest node to

        Returns:
            nearest_node: Nearest node in the tree

        Student Task:
        ------------
        1. Initialize variables for tracking nearest node and minimum distance
        2. Iterate through node_list
        3. Calculate distance to each node
        4. Update nearest node if current distance is smaller
        5. Return the nearest node found
        """
        nearest_node = None
        min_distance = float("inf")
        # YOUR CODE HERE
        # gonna just loop thru em all i guess
        # oop gotta remove the current node
        node_list_without_node = [n for n in self.node_list if n is not node]
        for neighbor_node in node_list_without_node:
            total_distance = self.calc_weighted_distance(node, neighbor_node)
            if total_distance < min_distance: # gotta set new minimums
                min_distance = total_distance
                nearest_node = neighbor_node


        return nearest_node

    def check_collision(
        self, node: "RRT.Node"
    ) -> bool:
        """
        Check if the node (and its path to parent) collides with any obstacles.

        Args:
            node: Node to check for collision
            obstacle_list: List of obstacles [(x, y, radius), ...]

        Returns:
            collision_free: True if node is collision-free, False otherwise

        Student Task:
        ------------
        1. For each obstacle:
           - Calculate distance from node to obstacle center
           - Check if distance is less than obstacle radius + robot_radius
        2. Return True if no collisions, False otherwise
        """
        # YOUR CODE GOES HERE
        # gotta discretize along the path
        # oh wait! my new path part property does that!!

        points_to_check = node.path

        for point in points_to_check:

            gx, gy = self.occupancy_grid.world_to_grid(point[0], point[1])
            # bounds check
            if gx < 0 or gy < 0 or gx >= self.occupancy_grid.image.shape[1] or gy >= self.occupancy_grid.image.shape[0]:
                return False
            if self.occupancy_grid.image[gy, gx]:
                return False
        
        return True
    
    
    def shortcut_path(self, path, max_iters=100):
        path = list(path)
        for _ in range(max_iters):
            if len(path) < 3:
                break
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            # temporarily reparent to test straight-line collision
            test_node = self.Node(path[j].x, path[j].y, path[j].psi)
            test_node.parent = path[i]
            if self.check_collision(test_node):
                path = path[:i+1] + [test_node] + path[j+1:]
        return path
    
    def densify_path(self, path, spacing=0.1):
        """Insert evenly-spaced nodes between consecutive path nodes.
        
        Args:
            path: list of nodes (any order — output preserves input order)
            spacing: target distance between samples in meters
        """
        if len(path) < 2:
            return list(path)
        
        dense = [path[0]]
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            dx, dy = b.x - a.x, b.y - a.y
            seg_len = math.hypot(dx, dy)
            n_steps = max(1, int(np.ceil(seg_len / spacing)))
            
            # interpolate heading too — handle wrap with shortest angular path
            dpsi = ((b.psi - a.psi + 180) % 360) - 180
            
            for k in range(1, n_steps + 1):
                t = k / n_steps
                new_node = self.Node(
                    a.x + t * dx,
                    a.y + t * dy,
                    a.psi + t * dpsi,
                )
                new_node.parent = a if k == 1 else dense[-1]
                dense.append(new_node)
        
        return dense


    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        
        # Plot the occupancy grid image as background
        extent = [
            self.occupancy_grid.origin[0],
            self.occupancy_grid.origin[0] + self.occupancy_grid.image.shape[1] * self.occupancy_grid.resolution,
            self.occupancy_grid.origin[1],
            self.occupancy_grid.origin[1] + self.occupancy_grid.image.shape[0] * self.occupancy_grid.resolution
        ]
        plt.imshow(self.occupancy_grid.image, cmap='gray_r', origin='lower', extent=extent, alpha=0.7)
        
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        
        # Plot tree edges
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # Arrow length for heading visualization
        arrow_len = 1.0

        if hasattr(self, 'angles') and hasattr(self, 'center'):
            cx, cy = self.center
            for theta in self.angles:
                # Convert center (pixel) to world coordinates for plotting
                wx, wy = self.occupancy_grid.grid_to_world(cx, cy)
                # Choose a reasonable length for the ray (in meters)
                ray_length = 5
                dx = ray_length * np.cos(theta)
                dy = ray_length * np.sin(theta)
                plt.plot([wx, wx + dx], [wy, wy + dy], '--', color='magenta', alpha=0.5)
                # Optionally, label the angle
                plt.text(wx + dx, wy + dy, f"{np.rad2deg(theta):.0f}°", color='magenta', fontsize=8)
            
        # Plot start node with heading arrow
        plt.plot(self.start.x, self.start.y, "xr", markersize=10)
        start_dx = arrow_len * np.cos(np.deg2rad(self.start.psi))
        start_dy = arrow_len * np.sin(np.deg2rad(self.start.psi))
        plt.arrow(self.start.x, self.start.y, start_dx, start_dy, 
                  head_width=0.3, head_length=0.2, fc='red', ec='red')
        plt.text(self.start.x, self.start.y + 0.5, f'Start psi={self.start.psi:.0f}°', color='red', fontsize=8)
        
        # Plot goal node with heading arrow
        plt.plot(self.end.x, self.end.y, "xb", markersize=10)
        end_dx = arrow_len * np.cos(np.deg2rad(self.end.psi))
        end_dy = arrow_len * np.sin(np.deg2rad(self.end.psi))
        plt.arrow(self.end.x, self.end.y, end_dx, end_dy,
                  head_width=0.3, head_length=0.2, fc='blue', ec='blue')
        plt.text(self.end.x, self.end.y + 0.5, f'Goal psi={self.end.psi:.0f}°', color='blue', fontsize=8)
        
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

def point_side_of_rectangle(pt3, box):
    # box: 4x2 array of rectangle corners, ordered clockwise
    # pt: (x, y)
    # Returns: side index (0=top, 1=right, 2=bottom, 3=left)
    pt = pt3[0:2]
    min_dist = float('inf')
    side_idx = -1
    for i in range(4):
        a = box[i]
        b = box[(i+1)%4]
        # Distance from pt to line segment ab
        ab = b - a
        ap = pt - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        dist = np.linalg.norm(pt - closest)
        if dist < min_dist:
            min_dist = dist
            side_idx = i
    return side_idx

def angle_from_vertical(pt, center):
    dx = pt[0] - center[0]
    dy = center[1] - pt[1]  # y axis is inverted in image coordinates
    angle = np.arctan2(dx, dy)  # 0 is up, positive is to the right
    return angle - 20

def min_dist_to_box(pt, box):
    # pt: (x, y)
    min_dist = float('inf')
    for i in range(4):
        a = box[i]
        b = box[(i+1)%4]
        ab = b - a
        ap = pt - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        dist = np.linalg.norm(pt - closest)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def angle_between(a, b, c):
    a = np.mod(a, 2*np.pi)
    b = np.mod(b, 2*np.pi)
    c = np.mod(c, 2*np.pi)
    if a < b:
        return a <= c <= b
    else:
        return c >= a or c <= b


def main():
    
    # load in the occupancy grid
    with open(r"src\Autonomy\rrt_planning\oh_my.yaml") as f:
        map_meta = yaml.safe_load(f)

    resolution = map_meta['resolution']
    origin = map_meta['origin']  # [x, y, theta]
    image_path = map_meta['image']

    
    image_file = os.path.basename(image_path)
    image_full_path = os.path.join("src", "Autonomy", "rrt_planning", image_file)
    raw = np.array(Image.open(image_full_path))
    grid = np.flipud(raw < 50) 

    # inflate walls

    grid = OccupancyGrid(origin, resolution, grid)

    plt.imshow(grid.image, cmap='gray', origin='lower')
    plt.title("White = free, Black = obstacle")
    plt.show()
    # Initialize plot

    img_uint8 = (grid.image.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    cv2.drawContours(img_color, [box], 0, (0, 255, 0), 2)
    

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = np.array([cx, cy])

    num_samples = 7
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    sampled_points = []

    free_space = (img_uint8 == 0).astype(np.uint8)

    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

    for theta in angles:
        direction = np.array([np.cos(theta), np.sin(theta)])
        max_dist = 0
        best_point = None
        for pt in cnt[:, 0, :]:
            vec = pt - center
            angle = np.arctan2(vec[1], vec[0])
            if np.isclose(np.mod(angle-theta, 2*np.pi), 0, atol=0.1):
                dist = np.linalg.norm(vec)
                if dist > max_dist:
                    max_dist = dist
                    candidate = np.zeros((3,))
                    candidate[0:2] = pt
                    candidate[0] += 15 * np.cos(theta)
                    candidate[1] += 15 * np.sin(theta)
                    if 0 <= candidate[1] < dist_transform.shape[0] and 0 <= candidate[0] < dist_transform.shape[1]:
                        if dist_transform[int(candidate[1]), int(candidate[0])] >= 10:
                            best_point = candidate

                            ## detect side of the rectangle for yaw
                            side = point_side_of_rectangle(best_point, box)
                            side_to_angle = {0: 90, 1: 180, 2: -90, 3: 0}
                            best_point[2] = side_to_angle[side]
        if best_point is not None:
            sampled_points.append(best_point)
    
    for pt in sampled_points:
        cv2.circle(img_color, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

    sampled_points = sorted(
    sampled_points,
    key=lambda pt: angle_from_vertical(pt, center),
    reverse=True)

    cv2.imshow('Detected Rectangles', img_color)
    cv2.waitKey(0)

    # get current pose 
    pose = np.array([0, 0, 0])

    # goal... 
    # we want it to return to origininal position?
    goal = deepcopy(pose)
    goals = sampled_points 

    # Plot all goal nodes with headings
    plt.figure(figsize=(10, 10))
    extent = [
        grid.origin[0],
        grid.origin[0] + grid.image.shape[1] * grid.resolution,
        grid.origin[1],
        grid.origin[1] + grid.image.shape[0] * grid.resolution
    ]
    plt.imshow(grid.image, cmap='gray_r', origin='lower', extent=extent, alpha=0.7)
    
    arrow_len = 1.5
    colors = ['green', 'blue', 'purple', 'orange', 'red']
    
    # Plot start position
    plt.plot(pose[0], pose[1], 'o', color='black', markersize=12, label='Start')
    start_dx = arrow_len * np.cos(np.deg2rad(pose[2]))
    start_dy = arrow_len * np.sin(np.deg2rad(pose[2]))
    plt.arrow(pose[0], pose[1], start_dx, start_dy,
              head_width=0.4, head_length=0.3, fc='black', ec='black')
    plt.text(pose[0] + 0.5, pose[1] + 0.5, f'Start\npsi={pose[2]:.0f}°', fontsize=8)
    
    # Plot all goal nodes
    for i, g in enumerate(goals):
        # Convert pixel to world coordinates
        gx, gy = int(g[0]), int(g[1])
        wx, wy = grid.grid_to_world(gx, gy)
        g[0] = wx
        g[1] = wy
        plt.plot(wx, wy, 'x', color=colors[i%len(colors)], markersize=12, markeredgewidth=3)
        dx = arrow_len * np.cos(np.deg2rad(g[2]))
        dy = arrow_len * np.sin(np.deg2rad(g[2]))
        plt.arrow(wx, wy, dx, dy,
                  head_width=0.4, head_length=0.3, fc=colors[i%len(colors)], ec=colors[i%len(colors)])
        plt.text(wx + 0.5, wy + 0.5, f'Goal {i+1}\npsi={g[2]:.0f}°', color=colors[i%len(colors)], fontsize=8)
    goals.append(np.array([0,0,0]))
    plt.axis("equal")
    plt.grid(True)
    plt.title("All Goal Nodes with Headings")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

    # workspace dims
    workspace_bounds = [np.array([15, 9, -180]), np.array([-25.0, -26.2, 180])]
                        
    total_path = []
    for goal in goals:


        rrt = RRT(
            start=pose,
            goal=goal,
            rand_area=workspace_bounds,
            occupancy_grid=grid,
            expand_dis = 0.3,
            max_iter=20000,
            max_steer=17.0,
            box = box
        )

        rrt.angles = angles
        rrt.center = center

        # Run planning
        nodes = rrt.planning(animation=False)

        if nodes is None:
            print("Cannot find path")
            continue

        print("found path!!")
        nodes = rrt.shortcut_path(nodes)
        #nodes = rrt.densify_path(nodes)
        nodes = nodes[::-1]

        # skip the first node only if it duplicates the previous segment's end
        if total_path:
            nodes = nodes[1:]

        total_path.extend(nodes)
        pose = np.array([total_path[-1].x, total_path[-1].y, total_path[-1].psi])

    # Export path to CSV using numpy
    path_data = []
    for node in total_path:
        path_data.append([node.x, node.y, node.psi])

    
    np.savetxt('ohmy_big_path.csv', path_data, delimiter=',', header='x,y,psi', comments='', fmt='%.6f')

    # Plot final path with color gradient
    plt.figure(figsize=(12, 10))
    extent = [
        grid.origin[0],
        grid.origin[0] + grid.image.shape[1] * grid.resolution,
        grid.origin[1],
        grid.origin[1] + grid.image.shape[0] * grid.resolution
    ]
    plt.imshow(grid.image, cmap='gray_r', origin='lower', extent=extent, alpha=0.5)
    
    # Extract path coordinates
    path_x = [node.x for node in total_path]
    path_y = [node.y for node in total_path]
    
    # Plot path segments with color gradient
    n_points = len(total_path)
    cmap = plt.cm.viridis  # Can also try: plasma, inferno, magma, coolwarm, rainbow
    
    for i in range(n_points - 1):
        color = cmap(i / n_points)
        plt.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                 color=color, linewidth=2)
    
    # Add colorbar to show progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_points))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Path Progress (waypoint index)')
    
    # Plot start and end markers
    plt.plot(total_path[0].x, total_path[0].y, 'go', markersize=15, label='Start', zorder=5)
    plt.plot(total_path[-1].x, total_path[-1].y, 'r*', markersize=20, label='End', zorder=5)
    
    # Plot intermediate goals
    for i, g in enumerate(goals[:-1]):
        plt.plot(g[0], g[1], 'x', color='orange', markersize=12, markeredgewidth=3)
        plt.text(g[0] + 0.3, g[1] + 0.3, f'G{i+1}', color='orange', fontsize=10)
    
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title("Complete Path with Color Gradient (Start → End)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc='upper right')

    plt.ioff()  # Disable interactive mode
    plt.show()  # This will block until the window is closed

if __name__ == "__main__":
    main()  # Run standard RRT

