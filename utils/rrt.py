import numpy as np
import pylab as pl
import random
from tqdm import tqdm
from matplotlib.lines import Line2D


class Node:
    '''
    Class for sampled point. Collection of nodes form tree. Each node has one parent and multiple children 
    Attributes:
        x, y (int)           : Coordinates
        parent (Node)        : Connected upstream node
        children (list[Node]): Connected downstream nodes
        cost (float)         : Distance to reach the node from tree's root node
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        self.cost = 0.0

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]
    
    def __array__(self, dtype=None):
        return np.array([self.x, self.y], dtype=dtype)
    
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __le__(self, other):
        return (self.x, self.y) <= (other.x, other.y)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __gt__(self, other):
        return (self.x, self.y) > (other.x, other.y)

    def __ge__(self, other):
        return (self.x, self.y) >= (other.x, other.y)

    def __hash__(self):
        return id(self)

class RRT:
    def __init__(self, env, start, goal, expand_dist=1.0, goal_bias_percent=5, max_iter=1000):
        '''
        RRT path planner.
        Args:
            env (class)              : Environment object from utils.environment_2d
            start (tuple)            : Start point for path planning
            goal (tuple)             : Goal point for path planning
            expand_dist (float)      : Maximum distance to expand sampled nodes
            goal_bias_percent (float): Percentage of sampling goal
            max_iter(int)            : Maximum iteration of random node sampling
        '''
        self.env = env
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.expand_dist = expand_dist
        self.goal_bias_percent = goal_bias_percent
        self.max_iter = max_iter
        self.node_list = [] # node list to store all nodes from all trees
        self.bias_node = self.goal # biased node during sampling

        self.tree_a = [self.start] # store nodes
        self.root_a = self.start # root of tree
        self.color_a = 'orange' # for visualization
        self.edge_plots = {} # store edge plots
        self.path_plots = [] # store path plots

        # Used in bidirectional RRT
        self.tree_b = [self.goal]
        self.root_b = self.goal
        self.color_b = 'magenta'
        self.tree_connect = [] # Store connected node pairs, [node from start_tree, node from goal_tree] 

    def get_random_node(self):
        # Sample random node. Probability in sampling biased node e.g. goal, start (bidirectional RRT).
        if random.randint(1, 100) > self.goal_bias_percent:
            rnd_x = np.random.rand()*self.env.size_x
            rnd_y = np.random.rand()*self.env.size_y
        else:
            rnd_x, rnd_y = self.bias_node.x, self.bias_node.y
        return Node(rnd_x, rnd_y)

    def get_nearest_node(self, tree, rnd_node):
        # Get nearest node to random sampled node from a tree
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in tree] # Distance
        min_index = dlist.index(min(dlist)) # Get index for smallest distance
        return tree[min_index]

    def steer(self, from_node, to_node, extend_length, animate=False, color='orange'):
        '''
        Extend tree by connecting a node to new sampled node within specific distance if collision-free
        Args:
            from_node (Node)    : Parent node of new connection, nearest node to sampled node
            to_node (Node)      : Target node of new connection, random sampled
            extnd_length (float): Maximum distance of new connection
            animate (bool)      : If True, pause between plots for dynamic visualization
            color (str)         : Color for plotting connection line and new nodes, Matplotlib color

        Returns:
            new_node (Node/None): Child node of new connection, None if not collision-free 
        '''
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * np.cos(theta)
        new_node.y += min(extend_length, d) * np.sin(theta)

        if not self.env.check_collision_line(from_node, new_node):
            self.update_parent(new_node, from_node, rewire=False, animate=animate, color=color)
            return new_node
        return None

    def calc_distance_and_angle(self, from_node, to_node):
        # Return [distance, angle] between two nodes
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def get_nearby_nodes(self, tree, new_node):
        # Return list of nearby nodes
        nearby_nodes = []
        for node in tree:
            d, _ = self.calc_distance_and_angle(node, new_node)
            if node != new_node and d <= self.expand_dist:
                nearby_nodes.append(node)
        return nearby_nodes

    def update_parent(self, node, parent, rewire=False, animate=False, color='orange'):
        '''
        Update node's parent & cost, and respective nodes' children.
        Set rewire=True if node changes parent
        Visualize parent update/new node added/children added.
        '''
        if rewire:
            node.parent.children.remove(node)
            red_line, = pl.plot([node.parent.x, node.x], [node.parent.y, node.y], color='tomato', linewidth=2)
            if animate:
                pl.pause(0.01)
            self.edge_plots[(node.parent, node)].remove()
            red_line.remove()

        node.parent = parent
        parent.children.append(node)
        node.cost = parent.cost + self.calc_distance_and_angle(parent, node)[0]
        
        if rewire:
            green_line, = pl.plot([parent.x, node.x], [parent.y, node.y], color='springgreen', linewidth=2)
            if animate:
                pl.pause(0.01)
            green_line.remove()
            line, = pl.plot([parent.x, node.x], [parent.y, node.y], color=color, linewidth=0.5)
            self.edge_plots[(parent, node)] = line

        if not rewire:
            line, = pl.plot([parent.x, node.x], [parent.y, node.y], color=color, linewidth=0.5)
            self.edge_plots[(parent, node)] = line
            pl.plot(node.x, node.y, "o", color=color, markersize=2)
            if animate:
                pl.pause(0.01)

    def update_subsequent_cost(self, tree, node):
        # Update costs of downstream nodes 
        for child in node.children:
            d, _ = self.calc_distance_and_angle(node, child)
            child.cost = node.cost + d
            self.update_subsequent_cost(tree, child)

    def rewire(self, tree, node, animate=False, color='orange'):
        '''
        Check for better (lower cost) parent of node and if it is a better parent for nearby nodes (for RRT*)
        '''
        nearby_nodes = self.get_nearby_nodes(tree, node)
        # Rewire parent
        rewire_parent = False
        best_parent = node.parent
        min_cost = node.cost
        for parent in nearby_nodes:
            d, _ = self.calc_distance_and_angle(parent, node)
            if parent.cost + d < min_cost and not self.env.check_collision_line(parent, node):
                rewire_parent = True
                best_parent = parent
                min_cost = parent.cost + d
        if rewire_parent:
            self.update_parent(node, best_parent, rewire=True, animate=animate, color=color)
        # Rewire children
        for child in nearby_nodes:
            d, _ = self.calc_distance_and_angle(node, child)
            if node.cost + d < child.cost and not self.env.check_collision_line(node, child):
                self.update_parent(child, node, rewire=True, animate=animate, color=color)
        
        self.update_subsequent_cost(tree, node)

    
    def check_connection(self, node1, node2):
        '''
        Check connection of two nodes (longer distance allowed)
        Used to check goal reaching & trees connection (for bidirectional RRT)
        '''
        return self.calc_distance_and_angle(node1, node2)[0] <= 2*self.expand_dist \
        and not self.env.check_collision_line(node1, node2)
    
    def generate_path(self, last_node):
        # Return list of path nodes and total cost from root node and end node
        path = []
        path_cost = last_node.cost
        while last_node:
            path.append((last_node.x, last_node.y))
            last_node = last_node.parent
        return path[::-1], path_cost

    def plot_path(self, path):
        # Plot paths and return plots (Line2D) for erase
        plots = []
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dot, = pl.plot(x0, y0, "o", color="green", markersize=3)
            plots.append(dot)
            line, = pl.plot([x0, x1], [y0, y1], color='green', linewidth=1.5)
            plots.append(line)
        dot, = pl.plot(path[-1][0], path[-1][1], "o", color="green", markersize=3)
        plots.append(dot)
        return plots

    def planning(self, bidir=False, star=False, sample_max=False, animate=True):
        '''
        Main method to plan path.
        Args:
            bidir (bool)     : If True, bidirectional RRT is used
            star (bool)      : If True, rewiring technique of RRT* is used
            sample_max (bool): If True, sampling continues after path is found until self.max_iter is reached, to improve path
            animate (bool)   : If True, pause between plots for dynamic visualization
        '''
        best_path = None
        min_path_cost = float('inf')

        for _ in tqdm(range(self.max_iter)):
            rnd_node = self.get_random_node()
            nearest_node_a = self.get_nearest_node(self.tree_a, rnd_node)
            new_node_a = self.steer(nearest_node_a, rnd_node, self.expand_dist, animate=animate, color=self.color_a)

            if new_node_a:
                self.tree_a.append(new_node_a)

                if star:
                    self.rewire(self.tree_a, new_node_a, animate=animate, color=self.color_a)

                if not bidir:
                    if self.check_connection(new_node_a, self.goal):
                        if self.goal not in self.tree_a: # first goal reaching
                            self.update_parent(self.goal, new_node_a, animate=animate)
                            self.tree_a.append(self.goal)
                            path, path_cost = self.generate_path(self.goal)
                            self.path_plots = self.plot_path(path)
                            if not sample_max:
                                return path, path_cost
                            else:
                                best_path = path
                                min_path_cost = path_cost
                        elif new_node_a.cost + self.calc_distance_and_angle(new_node_a, self.goal)[0] < self.goal.cost:
                            self.update_parent(self.goal, new_node_a, rewire=True, animate=animate, color=self.color_a)

                    if best_path:
                        path, path_cost = self.generate_path(self.goal)
                        if path_cost < min_path_cost:
                            best_path = path
                            min_path_cost = path_cost
                            for plot in self.path_plots:
                                plot.remove()
                            self.path_plots = self.plot_path(best_path)

                else: # bidirectional
                    nearest_node_b = self.get_nearest_node(self.tree_b, new_node_a)
                    new_node_b = self.steer(nearest_node_b, new_node_a, self.expand_dist, animate=animate, color=self.color_b)

                    if new_node_b:
                        self.tree_b.append(new_node_b)

                        if star:
                            self.rewire(self.tree_b, new_node_b, animate=animate, color=self.color_b)

                        if self.check_connection(new_node_a, new_node_b):
                            if len(self.tree_connect) == 0: # first trees connection
                                if self.root_a == self.start:
                                    self.tree_connect.append([new_node_a, new_node_b])
                                    path_a, path_cost_a = self.generate_path(new_node_a)
                                    path_b, path_cost_b = self.generate_path(new_node_b)
                                    path_b = path_b[::-1]
                                else:
                                    self.tree_connect.append([new_node_b, new_node_a])
                                    path_a, path_cost_a = self.generate_path(new_node_b)
                                    path_b, path_cost_b = self.generate_path(new_node_a)
                                    path_b = path_b[::-1]     
                                path = path_a + path_b
                                path_cost = path_cost_a + path_cost_b + self.calc_distance_and_angle(new_node_a, new_node_b)[0]
                                self.path_plots = self.plot_path(path)       
                                if not sample_max:
                                    return path, path_cost
                                else:
                                    best_path = path
                                    min_path_cost = path_cost
                        
                            if self.root_a == self.start:
                                connect = [new_node_a, new_node_b]
                            else:
                                connect = [new_node_b, new_node_a]

                            if connect not in self.tree_connect: # Check for new trees connection
                                if self.root_a == self.start:
                                    self.tree_connect.append([new_node_a, new_node_b])
                                else:
                                    self.tree_connect.append([new_node_b, new_node_a])
                        
                        if best_path: # Check better path
                            for node_a, node_b in self.tree_connect:
                                path_a, path_cost_a = self.generate_path(node_a)
                                path_b, path_cost_b = self.generate_path(node_b)
                                path_b = path_b[::-1]
                                path = path_a + path_b
                                path_cost = path_cost_a + path_cost_b + self.calc_distance_and_angle(new_node_a, new_node_b)[0]
                                if path_cost < min_path_cost:
                                    best_path = path
                                    min_path_cost = path_cost
                                    for plot in self.path_plots:
                                        plot.remove()
                                    self.path_plots = self.plot_path(best_path)
                    
                    # Switch a, b for average growth
                    self.tree_a, self.tree_b = self.tree_b, self.tree_a
                    self.root_a, self.root_b = self.root_b, self.root_a
                    self.color_a, self.color_b = self.color_b, self.color_a
                    self.bias_node = self.start if self.bias_node == self.goal else self.goal
        
        return best_path, min_path_cost

    def get_node_list(self):
        # Add relation for trees connection
        for connect in self.tree_connect:
            if connect[0] != connect[1]:
                connect[0].children.append(connect[1])
                connect[1].children.append(connect[0])
        return self.tree_a + self.tree_b

