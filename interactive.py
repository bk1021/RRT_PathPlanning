import numpy as np
import pylab as pl
import signal
from matplotlib.lines import Line2D
from utils.environment_2d import Environment
from utils.rrt import RRT, Node
from utils.astar import Astar

def signal_handler(sig, frame):
    print("\nProgram interrupted! Exiting gracefully...")
    pl.close('all')
    exit(0)


class InteractivePathPlanner:
    def __init__(self, expand_dist=0.5, goal_bias_percent=0.0, max_iter=3000, bidir=False, star=False, sample_max=False, animate=True):
        np.random.seed(4)
        self.env = Environment(10, 6, 5)
        self.env.plot()

        self.fig = pl.gcf()
        self.fig.canvas.mpl_connect('close_event', self.on_close) # Close program when cross out window

        q = None
        while not q:
            q = self.env.random_query()

        x_start, y_start, x_goal, y_goal = q
        self.query_plots = self.env.plot_query(x_start, y_start, x_goal, y_goal)

        self.start = (x_start, y_start)
        self.goal = (x_goal, y_goal)
        self.rrt = RRT(self.env, self.start, self.goal, expand_dist, goal_bias_percent, max_iter)
        self.path, self.path_cost = self.rrt.planning(bidir, star, sample_max, animate)
        self.node_list = self.rrt.get_node_list()

        self.edges = []
        for node in self.node_list:
            neighbors = [node.parent] + node.children if node.parent else node.children
            self.edges.extend([(node, neighbor) for neighbor in neighbors])

        if self.path is None:
            print("No path found!")
        else:
            print("Path found!")
            print("Path cost:", self.path_cost)

        self.path_plots = self.rrt.path_plots # Update path plots

        self.no_path_text_message = None # Store 'no path' text plotted

        # Set up key event handling using pylab
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.run_event_loop()

    def on_key_press(self, event):
        if event.key == ' ':
            q = None
            while not q:
                q = self.env.random_query()

            x_start, y_start, x_goal, y_goal = q
            # Remove previous query & path plots
            for plot in self.query_plots:
                plot.remove()
            for plot in self.path_plots:
                plot.remove()
            self.path_plots = []
            if self.no_path_text_message:
                self.no_path_text_message.remove()
            self.no_path_text_message = None
            # Plot query
            self.query_plots = self.env.plot_query(x_start, y_start, x_goal, y_goal)

            self.start = (x_start, y_start)
            self.goal = (x_goal, y_goal)

            print(f"New Start: ({x_start:.4f}, {y_start:.4f}), New Goal: ({x_goal:.4f}, {y_goal:.4f})")
            # Find path using astar
            new_path = self.astar_search()
            if new_path:
                print("New path found!")
                # Plot new path
                self.path_plots = self.rrt.plot_path(new_path)
            else:
                print("No path found!")
                self.display_no_path_message()
                       
        elif event.key == 'escape':
            print("\nEscape key pressed. Exiting gracefully...")
            pl.close('all')
            exit(0)
    
    def on_close(self, event):
        exit(0)

    def astar_search(self):
        start_node = Node(self.start[0], self.start[1])
        goal_node = Node(self.goal[0], self.goal[1])
        start_neighbors = self.rrt.get_nearby_nodes(self.node_list, start_node)
        goal_neighbors = self.rrt.get_nearby_nodes(self.node_list, goal_node)

        node_list = self.node_list + [start_node, goal_node]
        edges = self.edges.copy()
        for neighbor in start_neighbors:
            if self.rrt.check_connection(start_node, neighbor):
                edges.extend([(start_node, neighbor), (neighbor, start_node)])
        for neighbor in goal_neighbors:
            if self.rrt.check_connection(goal_node, neighbor):
                edges.extend([(goal_node, neighbor), (neighbor, goal_node)])
        if self.rrt.check_connection(start_node, goal_node):
            edges.append((start_node, goal_node))

        astar = Astar(node_list, edges, start_node, goal_node)
        path = astar.a_star()
        if path:
            return [(node.x, node.y) for node in path]
        return None
    
    def display_no_path_message(self):
        ax = pl.gca()
        self.no_path_text_message = ax.text(
            0.5, 0.5, "No path found!", fontsize=14, color='red',
            ha='center', va='center', transform=ax.transAxes, fontweight='bold'
        )

    def run_event_loop(self):
        while True:
            pl.pause(0.01)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    planner = InteractivePathPlanner(bidir=False, star=False, sample_max=False, animate=True)