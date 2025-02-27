import numpy as np
import pylab as pl
import signal
import argparse
from utils.environment_2d import Environment
from utils.rrt import RRT

def signal_handler(sig, frame):
    print("\nProgram interrupted! Exiting gracefully...")
    pl.close('all')
    exit(0)

def on_close(event):
    exit(0)

def simulate_rrt(size_x=10, size_y=6, n_obs=5, random_seed=None, expand_dist=0.5, goal_bias_percent=0, 
                 max_iter=1000, bidir=False, star=False, sample_max=False, animate=True):
    if random_seed:
        np.random.seed(4)
    env = Environment(size_x, size_y, n_obs)
    env.plot()

    fig = pl.gcf()
    fig = fig.canvas.mpl_connect('close_event', on_close) # Close program when cross out window

    q = None
    while not q:
        q = env.random_query()

    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)

    start = (x_start, y_start)
    goal = (x_goal, y_goal)
    rrt = RRT(env, start, goal, expand_dist, goal_bias_percent, max_iter)
    path, path_cost = rrt.planning(bidir, star, sample_max, animate)

    if path is None:
        print("No path found!")
    else:
        print("Path found!")
        print("Path cost:", path_cost)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RRT Path Planning Simulation Arguments")
    parser.add_argument('-x', type=int, help="Environment X size", default=10)
    parser.add_argument('-y', type=int, help="Environment Y size", default=6)
    parser.add_argument('-o', type=int, help="Number of obstacles", default=5)
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--len', type=float, help="Max distance allowed for node expansion", default=0.5)
    parser.add_argument('--bias', type=float, help="Percentage of biased-sampled goal", default=0)
    parser.add_argument('--iter', type=int, help="Max sampling iteration", default=1000)
    parser.add_argument('--bidir', help="Use bidirectional RRT", action="store_true")
    parser.add_argument('--star', help="Use RRT*", action="store_true")
    parser.add_argument('--max', help="Sampling for max iteration", action="store_true")
    parser.add_argument('--ani', help="Disable animated visualization", action="store_false")

    arg = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    simulate_rrt(arg.x, arg.y, arg.o, arg.seed, arg.len, arg.bias, arg.iter, arg.bidir, arg.star, arg.max, arg.ani)
    pl.ioff()
    pl.show()