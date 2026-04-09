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

import matplotlib.pyplot as plt
import numpy as np
import yaml
from copy import deepcopy
from PIL import Image

show_animation = True


class OccupancyGrid():

    def __init__(self, origin, resolution, image):

        self.origin = origin
        self.resolution = resolution
        self.image = image

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
        max_steer: float = 15.0
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
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
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
        self.theta_weight = 0.001
        self.max_steer = max_steer

    # ADDED CODE! A lovely part property for the final path
    @property
    def final_path(self):

        total_path = []
        node = self.node_list[-1] # should be the end node
        while node.parent is not None: # only the start node should have a parent of none
            sub_path = node.path
            total_path.extend(sub_path)
            node = node.parent # move to the parent

        return total_path

    def get_random_node(self) -> "RRT.Node":
        """
        Sample a random node in the state space.

        The sampling should account for:
        - goal_sample_rate: Probability of sampling the goal node
        - min_rand/max_rand: Bounds of the sampling space

        Student Task:
        ------------
        1. With probability goal_sample_rate/100, return the goal node
        2. Otherwise, sample random position within min_rand/max_rand bounds
        4. Return new Node with sampled values

        Returns:
            Node: Randomly sampled node
        """

        is_goal = False
        random_num = random.random()
        if random_num < self.goal_sample_rate/100: # gotta divide by 100 to make range 0 to 1
            random_node = deepcopy(self.end) # imma deepcopy to be safe
            random_node.psi = random.uniform(self.min_rand[2], self.max_rand[2])
            is_goal = True
        else:
            # need a rando sample
            rand_x = random.uniform(self.min_rand[0], self.max_rand[0])
            rand_y = random.uniform(self.min_rand[1], self.max_rand[1])
            rand_psi = random.uniform(self.min_rand[2], self.max_rand[2])

            random_node = self.Node(rand_x, rand_y, rand_psi)

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
        self, animation: bool = True
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
            if animation and i % 50 == 0:
                self.draw_graph(rnd_node)
            # YOUR CODE GOES HERE
            # Check if goal is reachable from new node

            if proposed_node is not None:
                vec_to_goal = np.array([self.end.x - proposed_node.x, self.end.y - proposed_node.y]) 
                distance_to_goal = np.linalg.norm(vec_to_goal)
                
                if distance_to_goal < self.expand_dis:
                    
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

        if animation:
            # For standard RRT, plot straight lines
            path_x = [pnt[0] for pnt in final_path]
            path_y = [pnt[1] for pnt in final_path]
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

        plt.plot(self.start.x, self.start.y, "xr", markersize=10)
        plt.plot(self.end.x, self.end.y, "xb", markersize=10)
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
        
def main():
    
    # load in the occupancy grid
    with open(r"src\Autonomy\rrt_planning\test.yaml") as f:
        map_meta = yaml.safe_load(f)

    resolution = map_meta['resolution']
    origin = map_meta['origin']  # [x, y, theta]
    image_path = map_meta['image']

    raw = np.array(Image.open(image_path))
    grid = np.flipud(raw < 50) 

    # inflate walls

    grid = OccupancyGrid(origin, resolution, grid)

    plt.imshow(grid.image, cmap='gray', origin='lower')
    plt.title("White = free, Black = obstacle")
    plt.show()
    # Initialize plot

    # get current pose 
    pose = np.array([2, 2, 90])

    # goal... 
    # we want it to return to origininal position?
    goal = deepcopy(pose)
    goal = np.array([8, 8, 90])

    # workspace dims
    workspace_bounds = [np.array([0, 0, -180]), np.array([10, 10, 180])]



    rrt = RRT(
        start=pose,
        goal=goal,
        rand_area=workspace_bounds,
        occupancy_grid=grid,
        robot_radius=0.8,
        expand_dis = 0.3,
        max_iter=10000
    )

    # Run planning
    path = rrt.planning(animation=True)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    plt.ioff()  # Disable interactive mode
    plt.show()  # This will block until the window is closed

if __name__ == "__main__":
    main()  # Run standard RRT

