# MISD
This is a PyTorch implementation for our paper: "Efficient Hierarchical Reinforcement Learning via Mutual Information Constrained Subgoal Discovery".

Our code is based on official implementation of [HIGL](https://github.com/junsu-kim97/HIGL) (NeurIPS 2021).
## Installation
```
conda create -n misd python=3.6
conda activate misd
./install_all.sh
```
Also, to run the MuJoCo experiments, a license is required (see [here](https://www.roboti.us/license.html)).

## Usage

- Spiral
```
./scripts/misd_spiral_maze.sh ${timesteps} ${gpu} ${seed}
./scripts/misd_spiral_maze.sh 2e6 0 2
```

- Point Maze (U-shape)
```
./scripts/misd_point_maze_u.sh ${timesteps} ${gpu} ${seed}
./scripts/misd_point_maze_u.sh 2e6 0 2
```

- Ant Maze (U-shape)
```
./scripts/misd_ant_maze_u.sh ${timesteps} ${gpu} ${seed}
./scripts/misd_ant_maze_u.sh 2e6 0 2
```

- Ant Maze (S-shape)
```
./scripts/misd_ant_maze_s.sh ${timesteps} ${gpu} ${seed}
./scripts/misd_ant_maze_s.sh 2e6 0 2
```

- Ant TwoRooms
```
./scripts/misd_ant_two_rooms.sh ${timesteps} ${gpu} ${seed}
./scripts/misd_ant_two_rooms.sh 2e6 0 2
```






