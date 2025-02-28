# RRT_PathPlanning
Rapidly-exploring Random Trees (RRT) is an algorithm for path planning and exploration. It is useful for navigating spaces in complex environment. [Bidirectional RRT](#bidirectional-rrt) & [RRT*](#rrt*) are variation/extended versions of regular RRT. RRT repeatly perform the steps below until a path is found from the start to goal or a specified number of iterations is reached.
1. Randomly samples point in the space
2. Extends tree towards it
3. Checks for collisions

## RRT
<p align="center">
    <img src="https://i.postimg.cc/6pDhJW3c/rrt.gif" alt="Alt Text" width="600" height="400">
</p>

## Bidirectional RRT
Bidirectional RRT grows two tree, one from start and one from goal, simultaneosly.
<p align="center">
    <img src="https://i.postimg.cc/3JwPxP4s/bidir-rrt.gif" alt="Alt Text" width="600" height="400">
</p>

## RRT*
RRT* optimizes the path by performing rewiring during node sampling. Rewiring of sampled node ensure it is connected to a parent & children of lowest cost. RRT* can improve the path with more nodes sampled.
<p align="center">
    <img src="https://i.postimg.cc/7LhzPFpQ/star.gif" alt="Alt Text" width="600" height="400">
</p>

## Astar
The nodes sampled by RRT can be used by other searched-based path planning algorithm e.g. Astar, to obtain paths for other start-goal pairs
<p align="center">
    <img src="https://i.postimg.cc/bwLWbndF/astarsearch.gif" alt="Alt Text" width="600" height="400">
</p>

## Scripts
1. `rrt_simulate.py` run rrt on a random simulated environment returning the path & path cost, `pylab` plot visualization allowed, you can specify the arguments in command line as shown below.

```
usage: rrt_simulate.py [-h] [-x X] [-y Y] [-o O] [--seed SEED] [--len LEN] [--bias BIAS] [--iter ITER] [--bidir] [--star] [--max] [--ani]

RRT Path Planning Simulation Arguments

options:
  -h, --help   show this help message and exit
  -x X         Environment X size
  -y Y         Environment Y size
  -o O         Number of obstacles
  --seed SEED  Random seed
  --len LEN    Max distance allowed for node expansion
  --bias BIAS  Percentage of biased-sampled goal
  --iter ITER  Max sampling iteration
  --bidir      Use bidirectional RRT
  --star       Use RRT*
  --max        Sampling for max iteration
  --ani        Disable animated visualization
```

2. `interactive.py` run rrt first, like `rrt_simulate.py`. After rrt finish planning the first path query, you can press `spacebar` key to change a random path query, the program use astar to find the path solution, the process are visualized on the `pylab` plot. Exit by pressing `Ctrl+C` or cross out the plot window.

