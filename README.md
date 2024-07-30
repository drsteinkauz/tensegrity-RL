# tensegrity-RL
A repository for training a tensegrity robot to move using reinforcement learning.

Environment: A tensegrity robot moving on either a flat plane or an uneven surface. The tensegrity robot consists of three rigid bars connected together by 6 actuated tendons and 3 unacuated tendons. 

Observations: The angular position of each rigid bar, the angular velocity of each rigid bar, the linear velocity of each rigid bar, and the length of each tendon. 

Actions: 6 actuated tendons. Actuating a tendon causes it to change length. The actuator operates in the range of -0.45 to -0.15.

Reward: (change_in_position_or_heading * desired_direction)/dt. The reward is based on the linear velocity (when the goal is to move in a straight line) or angular velocity (when the goal is to turn in place) of the tensegrity robot. The desired direction is either 1 or -1, indicating if the robot should learn to move forward (1) or backward (-1) when moving straight, or turn counterclockwise (1) or clockwise (-1) when turning in place. dt is the change in time between actions.

## Getting started 

##### Create Conda environment:

It is recommended that you create a conda environment. 

```
conda create -n "tensegrity" python=3.8.10
conda activate tensegrity
```

##### Install MuJoCo:
1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

If you want to specify a nonstandard location for the package,
use the env variable `MUJOCO_PY_MUJOCO_PATH`.

##### Installing packages:

Navigate to this repo's directory
```
pip install -r requirements.txt
```

##### Installing custom environment:

To install the custom tensegrity environment so that it can be used as a Gym environment,
```
cd tensegrity_env
pip install -e .
cd ..
```

If you make any changes to the tensegrity environment (the tensegrity_env.py or any file in the tensegrit_env directory), you must reinstall the environment.
```
cd tensegrity_env
rm -r tensegrity_env.egg-info
pip install -e .
cd ..

```

## Commands to train and test the tensegrity

The ```run.py ``` is the main Python file that will be run. Below are the following arguements for this file.

Arguement      | Default | Description
------------------------| ------------- | ----------
--train  | no default | Either --train or --test must be specified. --train is for training the RL model and --test is for viewing the results of the model. After --test, the path to the model to be tested must be given.
--test  | no default | Either --train or --test must be specified. --train is for training the RL model and --test is for viewing the results of the model. After --test, the path to the model to be tested must be given.
--env_xml | "3prism_jonathan_steady_side.xml" | The name of the xml file for the MuJoCo environment. This xml file must be located in the same directory as ```run.py```
--sb3_algo | "SAC" | The Stable Baselines3 RL algorithm. Options are "SAC", "TD3", "A2C", or "PPO".
--desired_action | "straight" | What goal the RL model is trying to accomplish. Options are "straight" or "turn"
--desired_direction | 1 | The direction the RL model is trying to move the tensegrity. Options are 1 (forward or counterclockwise) or -1 (backward or clockwise)
--delay | 1 | How many steps to take in the environment before updating the critic. Options are 1, 10, or 100, but 1 worked best
--log_dir | "logs" | The directory where the training logs will be saved
--model_dir | "models" | The directory where the trained models will be saved
--saved_data_dir | "saved_data" | The directory where the data collected when testing the model will be saved (tendon length, contact with ground, actions)
--simulation_seconds | 30 | How many seconds the simulation should be run when testing a model


To train an RL model using SAC that moves the tensegirty forward:
```
python3 run.py --train --desired_action straight --desired_direction 1 --delay 10 --model_dir models_forward --log_dir logs_forward
```

To train an RL model using SAC that moves the tensegirty backward:
```
python3 run.py --train --desired_action straight --desired_direction -1 --delay 10 --model_dir models_backward --log_dir logs_backward
```

To train an RL model using SAC that turns the tensegrity counterclockwise:
```
python3 run.py --train --desired_action turn --desired_direction 1 --delay 10 --terminate_when_unhealthy no --model_dir models_ccw --log_dir logs_ccw
```

To train an RL model using SAC that turns the tensegrity clockwise:
```
python3 run.py --train --desired_action turn --desired_direction -1 --delay 10 --terminate_when_unhealthy no --model_dir models_cw --log_dir logs_cw
```

To test an RL model

```
python3 run.py --test ./forward_models/SAC_1875000.zip --simulation_seconds 30
```


