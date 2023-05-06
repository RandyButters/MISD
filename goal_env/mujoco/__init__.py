from gym.envs.registration import register
import gym

robots = ['Point', 'Ant']
task_types = ['Maze', 'Maze1', 'MazeL', 'Push', 'Fall', 'Block', 'BlockMaze', 'MazeW', 'MazeS', 'MazeB', 'FourRooms', 'TwoRooms']
all_name = [x + y for x in robots for y in task_types]
for name_t in all_name:
    for Test in ['', 'Test']:
        # The episode length for train is 200
        max_timestep = 200
        random_start = True
        if name_t[-4:] == 'Maze' or name_t[-4:] == 'aze1':
            goal_args = [[-4, -4], [20, 20]]
        if name_t[-9:] == 'FourRooms':
            goal_args = [[-3, -3], [15, 15]]
        if Test == 'Test':
            goal_args = [[0.0, 16.0], [1e-3,16 + 1e-3]]
            random_start = False
            # The episode length for test is 500
            max_timestep = 500

        register(
            id=name_t + Test + '-v0',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 8, 'random_start':random_start},
            max_episode_steps=max_timestep,
        )

        # v1 is the one we use in the main paper
        register(
            id=name_t + Test + '-v1',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 4, 'random_start':random_start},
            max_episode_steps=max_timestep,
        )

        register(
            id=name_t + Test + '-v2',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 2, 'random_start':random_start},
            max_episode_steps=max_timestep,
        )
        register(
            id=name_t + Test + '-v3',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 1, 'random_start':random_start},
            max_episode_steps=max_timestep,
        )

register(
    id='Spiral-v0',
    entry_point='goal_env.mujoco.spiral:SpiralMazeEnv',
    max_episode_steps=200
)

register(
    id='NineRooms-v0',
    entry_point='goal_env.mujoco.spiral:SpiralMazeEnv',
    max_episode_steps=200
)

register(
    id='Reacher3D-v0',
    entry_point='goal_env.mujoco.create_fetch_env:create_fetch_env',
    kwargs={'env_name': 'Reacher3D-v0'},
    max_episode_steps=100
)

register(
    id='Pusher-v0',
    entry_point='goal_env.mujoco.create_fetch_env:create_fetch_env',
    kwargs={'env_name': 'Pusher-v0'},
    max_episode_steps=100
)