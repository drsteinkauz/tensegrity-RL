# to create a python package, move to the custom_tensegrity top folder and 
# pip install -e .


import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
from scipy.spatial.transform import Rotation
from collections import deque
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class tr_env(MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is based on the tensegrity robot, which consists of 3 rigid bars connected
    together by 9 tendonds. Tendons 0 through 5 are short tendons whose length can be changed by
    motors. Tendons 6, 7, and 8 are long tendons that cannot be accuated and act as springs. 
    
    The Mujoco xml file contains 3 bodies that represent the 3 rigid bars:
    r01_body, r23_body, and r45_body
    
    The Mujoco xml file contains 9 tendons
    td0: red endcap to light blue endcap, act_0: 
    td1: red endcap to dark blue endcap, act_1
    td2: light blue endcap to dark blue endcap, act_2
    td3: green endcap to pink endcap,   act_3
    td4: green endcap to yellow endcap,   act_4
    td5: pink endcap to yellow endcap,   act_5
    td6: light blue endcap to green endcap  
    td7: red endcap to yellow endcap
    td8: dark blue endcap to pink endcap


    ### Action Space
    The action space is a `Box(-0.45, 0.15, (6,), float32)`. 

    | Num | Action                           | Control Min | Control Max | Name (in XML file) | Tendon |
    | --- | ---------------------------------| ----------- | ----------- | -------------------| -----  | 
    | 0   | td0: right r23 to right r45      | -0.45       | 0.15        | act_0              | td0    |
    | 1   | td1: right r01 to right r23      | -0.45       | 0.15        | act_1              | td1    |
    | 2   | td2: right r01 to right r45      | -0.45       | 0.15        | act_2              | td2    |
    | 3   | td3: left r01 to left r23        | -0.45       | 0.15        | act_3              | td3    |
    | 4   | td4: left r01 to left r45        | -0.45       | 0.15        | act_4              | td4    |
    | 5   | left r23 to left r45             | -0.45       | 0.15        | act_5              | td5    | 

    ### Observation Space 
    Observations: 45 (with velocity) / 27 (without velocity)

    | Idx |       Observation       | Num | Unit           | Min  | Max |
    |-----|-------------------------|-----|----------------|------|-----|
    | 1   | relative position of s0 | 3   | length (m)     | -Inf | Inf |
    | 2   | relative position of s1 | 3   | length (m)     | -Inf | Inf |
    | 3   | relative position of s2 | 3   | length (m)     | -Inf | Inf |
    | 4   | relative position of s3 | 3   | length (m)     | -Inf | Inf |
    | 5   | relative position of s4 | 3   | length (m)     | -Inf | Inf |
    | 6   | relative position of s5 | 3   | length (m)     | -Inf | Inf |

    | 7   | velocity of s0          | 3   | velocity (m/s) | -Inf | Inf |
    | 8   | velocity of s1          | 3   | velocity (m/s) | -Inf | Inf |
    | 9   | velocity of s2          | 3   | velocity (m/s) | -Inf | Inf |
    | 10  | velocity of s3          | 3   | velocity (m/s) | -Inf | Inf |
    | 11  | velocity of s4          | 3   | velocity (m/s) | -Inf | Inf |
    | 12  | velocity of s5          | 3   | velocity (m/s) | -Inf | Inf |

    | 13  | td0 length              | 1   | length (m)     | -Inf | Inf |
    | 14  | td1 length              | 1   | length (m)     | -Inf | Inf |
    | 15  | td2 length              | 1   | length (m)     | -Inf | Inf |
    | 16  | td3 length              | 1   | length (m)     | -Inf | Inf |
    | 17  | td4 length              | 1   | length (m)     | -Inf | Inf |
    | 18  | td5 length              | 1   | length (m)     | -Inf | Inf |
    | 19  | td6 length              | 1   | length (m)     | -Inf | Inf |
    | 20  | td7 length              | 1   | length (m)     | -Inf | Inf |
    | 21  | td8 length              | 1   | length (m)     | -Inf | Inf |

    ### Rewards
    The reward consists of:

    - *forward_reward*: A reward of moving or turning in the desired direction which is measured as
    (change_in_desired_quality * desired_direction)/dt*. *dt* is the time
    between actions and is dependent on the `frame_skip` parameter (default is 5),
    where the frametime is 0.01 - making the default *dt = 5 * 0.01 = 0.05*.

    - *ctrl_cost*: A negative reward for penalising the tensegrity if it takes actions
    that are too large. It is measured as *`ctrl_cost_weight` * sum(action<sup>2</sup>)*
    where *`ctr_cost_weight`* is a parameter set for the control and has a default value of 0.5.

    - *contact_cost*: A negative reward for penalising the tensegrity if the external contact
    force is too large. It is calculated *`contact_cost_weight` * sum(clip(external contact
    force to `contact_force_range`)<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost* and `info` will also contain the individual reward terms.



    ### Arguments

    | Parameter               | Type       | Default      |Description                    |
    |-------------------------|------------|--------------|-------------------------------|
    | `xml_file`              | **str**    | `"3prism_jonathan_steady_side.xml"`  | Path to a MuJoCo model |
    | `ctrl_cost_weight`      | **float**  | `0.001`        | Weight for *ctrl_cost* term (see section on reward) |
    | `use_contact_forces`    | **bool**| `False`       | If the reward should be penalized for contact forces between the rigid bars and ground
    | `contact_cost_weight`   | **float**  | `5e-4`       | Weight for *contact_cost* term (see section on reward) |
    | `healthy_reward`        | **float**  | `0.1`          | Constant reward given if the tensegrity is "healthy" after timestep |
    | `terminate_when_unhealthy` | **bool**| `True`       | If true, issue a done signal if the z-coordinate of the torso is no longer in the `healthy_z_range` |
    | `contact_force_range`   | **tuple**  | `(-1.0, 1.0)`    | Contact forces are clipped to this range in the computation of *contact_cost* |
    | `reset_noise_scale`     | **float**  | `0.0`        | Scale of random perturbations of initial position and velocity (parameter has been replaced by the following 4 parameters) |
    | `min_reset_heading`     | **float**  | `0.0`        | The minimum heading the tensegrity can have after being reset |
    | `max_reset_heading`     | **float**  | `2*np.pi`        | The maximum heading the tensegrity can have after being reset |
    | `tendon_reset_mean`     | **float**  | `0.15`         | The mean tendon length after the tensegrity has been reset  |
    | `tendon_reset_stdev`     | **float**  | `0.1`        | The standard deviation tendon length after the tensegrity has been reset  |
    | `tendon_max_length`     | **float**  | `0.15`         | The maximum tendon length after the tensegrity has been reset  |
    | `tendon_min_length`     | **float**  | `-0.45`        | The minimum tendon length after the tensegrity has been reset  |
    | `desired_action`     | **str**  | `"straight"`        | The desired action which the RL model should learn, either straight or turn  |
    | `desired_direction`     | **float**  | `1`        | The desired direction the tensegrity should move  |
    | `reward_delay_seconds`     | **float**  | `0.5`        | Just when turning, the delay between the old heading and the current heading when calculating the change in heading |
    | 'contact_with_self_penalty' | **float**  | `0.0`        | The penalty multiplied by the total contact between bars, subtracted from the reward. |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file=os.path.join(os.getcwd(),"3prism_jonathan_steady_side.xml"),
        use_contact_forces=False,
        use_cap_velocity=True,
        use_obs_noise = False,
        use_cap_size_noise = False,
        terminate_when_unhealthy=True,
        is_test = False,
        desired_action = "straight",
        desired_direction = 1,
        ctrl_cost_weight=0.01,
        contact_cost_weight=5e-4,
        healthy_reward=0.1, 
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.0, # reset noise is handled in the following 4 variables
        min_reset_heading = 0.0,
        max_reset_heading = 2*np.pi,
        tendon_reset_mean = 0.15,
        tendon_reset_stdev = 1.1,
        tendon_max_length = 0.15,
        tendon_min_length = -0.45,
        reward_delay_seconds = 0.02, # 0.5
        contact_with_self_penalty = 0.0,
        obs_noise_tendon_stdev = 0.02,
        obs_noise_cap_pos_stdev = 0.05,
        cap_size_noise_range = (0.04, 0.09),
        way_pts_range = (2.5, 3.5),
        way_pts_angle_range = (-np.pi/6, np.pi/6),
        threshold_waypt = 0.05,
        ditch_reward_max=300,
        ditch_reward_stdev=0.15,
        waypt_reward_amplitude=100,
        waypt_reward_stdev=0.10,
        yaw_reward_weight=1,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            use_contact_forces,
            use_cap_velocity,
            use_obs_noise,
            use_cap_size_noise,
            terminate_when_unhealthy,
            is_test,
            desired_action,
            desired_direction,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            contact_force_range,
            reset_noise_scale,
            min_reset_heading,
            max_reset_heading,
            tendon_reset_mean,
            tendon_reset_stdev,
            tendon_max_length,
            tendon_min_length,
            reward_delay_seconds,
            contact_with_self_penalty,
            obs_noise_tendon_stdev,
            obs_noise_cap_pos_stdev,
            cap_size_noise_range,
            way_pts_range,
            way_pts_angle_range,
            threshold_waypt,
            ditch_reward_max,
            ditch_reward_stdev,
            waypt_reward_amplitude,
            waypt_reward_stdev,
            yaw_reward_weight,
            **kwargs
        )
        self._x_velocity = 1
        self._y_velocity = 1
        self._is_test = is_test
        self._desired_action = desired_action
        self._desired_direction = desired_direction
        self._reset_psi = 0
        self._psi_wrap_around_count = 0
        self._use_cap_velocity = use_cap_velocity
        
        self._oripoint = np.array([0.0, 0.0])
        self._waypt_range = way_pts_range
        self._waypt_angle_range = way_pts_angle_range
        self._threshold_waypt = threshold_waypt
        self._ditch_reward_max = ditch_reward_max
        self._ditch_reward_stdev = ditch_reward_stdev
        self._waypt_reward_amplitude = waypt_reward_amplitude
        self._waypt_reward_stdev = waypt_reward_stdev
        # self._tracking_fwd_weight = tracking_fwd_weight
        self._yaw_reward_weight = yaw_reward_weight
        self._waypt = np.array([])


        self._use_obs_noise = use_obs_noise
        self._obs_noise_tendon_stdev = obs_noise_tendon_stdev
        self._obs_noise_cap_pos_stdev = obs_noise_cap_pos_stdev
        self._use_cap_size_noise = use_cap_size_noise
        self._cap_size_noise_range = cap_size_noise_range

        self._min_reset_heading = min_reset_heading
        self._max_reset_heading = max_reset_heading
        self._tendon_reset_mean = tendon_reset_mean
        self._tendon_reset_stdev = tendon_reset_stdev
        self._tendon_max_length = tendon_max_length
        self._tendon_min_length = tendon_min_length

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        if self._desired_action == "turn":
            self._contact_force_range = (-1000.0, 1000.0)
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces

        self._contact_with_self_penalty = contact_with_self_penalty

        obs_shape = 27
        if use_contact_forces:
            obs_shape += 84
        if use_cap_velocity:
            obs_shape += 18
        if desired_action == "tracking" or desired_action == "aiming":
            obs_shape += 3 # cmd lin_vel * 2 + ang_vel * 1

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        frame_skip = 20
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space=observation_space, **kwargs
        )
        self._reward_delay_steps = int(reward_delay_seconds/self.dt)
        self._heading_buffer = deque()

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action, tendon_length_6):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action + 0.5 - tendon_length_6)) # 0.5 is the initial spring length for 6 tendons
        # control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        if self._desired_action == "turn" or self._desired_action == "aiming":
            bar_speeds = np.abs(state[21:])
            min_velocity = 0.1
            is_healthy = np.isfinite(state).all() and (np.any(bar_speeds > min_velocity) )    

        else: #self._desired_action == "straight" or self._desired_action == "tracking":
            min_velocity = 0.0001
            is_healthy = np.isfinite(state).all() and ((self._x_velocity > min_velocity or self._x_velocity < -min_velocity) \
                                                        or (self._y_velocity > min_velocity or self._y_velocity < -min_velocity) )
            
        
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        
        xy_position_before = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3

        self.do_simulation(action, self.frame_skip)
        xy_position_after = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        self._x_velocity, self._y_velocity = xy_velocity

        x_position_before, y_position_before = xy_position_before
        x_position_after, y_position_after = xy_position_after

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_after = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_after = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3

        orientation_vector_after = left_COM_after - right_COM_after
        psi_after = np.arctan2(-orientation_vector_after[0], orientation_vector_after[1])

        tendon_length = np.array(self.data.ten_length)
        tendon_length_6 = tendon_length[:6]


        if self._desired_action == "turn":
            self._heading_buffer.append(psi_after)
            if len(self._heading_buffer) > self._reward_delay_steps:
                old_psi = self._heading_buffer.popleft()

                # unless the tensegrity is rotating faster than pi /(self.dt*self._reward_delay_steps) rad/s
                # then this situation means that the tensegrity rolled from pi to -pi, and the delta should be positive
                if psi_after < -np.pi/2 and old_psi > np.pi/2: 
                    psi_after = 2*np.pi + psi_after
                # unless the tensegrity is rotating faster than pi /(self.dt*self._reward_delay_steps) rad/s
                # then this situation means that the tensegrity rolled from -pi to pi, and the delta should be negative
                elif psi_after > np.pi/2 and old_psi < -np.pi/2:
                    psi_after = -2*np.pi + psi_after
                delta_psi = (psi_after - old_psi) / (self.dt*self._reward_delay_steps)
                forward_reward = delta_psi * self._desired_direction 
                costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            else:
                forward_reward = 0
                costs = ctrl_cost =  0
                delta_psi = 0
            
            if self._terminate_when_unhealthy:
                healthy_reward = self.healthy_reward
            else:
                healthy_reward = 0
            
            terminated = self.terminated  


        elif self._desired_action == "straight":
            
            psi_movement = np.arctan2(y_position_after-y_position_before, x_position_after-x_position_before)

            psi_diff = np.abs(psi_movement-self._reset_psi)

            forward_reward = self._desired_direction*\
                                (np.sqrt((x_position_after-x_position_before)**2 + \
                                        (y_position_after - y_position_before)**2) *\
                                np.cos(psi_diff)/ self.dt)

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            if self._terminate_when_unhealthy:
                healthy_reward = self.healthy_reward
            else:
                healthy_reward = 0
            
            terminated = self.terminated

        elif self._desired_action == "aiming":
            target_direction = self._waypt - xy_position_before
            target_direction = target_direction / np.linalg.norm(target_direction)
            target_psi = np.arctan2(target_direction[1], target_direction[0])
            new_psi_rbt_tgt = self._angle_normalize(target_psi - psi_after)
            self._heading_buffer.append(new_psi_rbt_tgt)
            if len(self._heading_buffer) > self._reward_delay_steps:
                old_psi_rbt_tgt = self._heading_buffer.popleft()
                delta_psi = -(np.abs(new_psi_rbt_tgt) - np.abs(old_psi_rbt_tgt)) / (self.dt*self._reward_delay_steps)
                forward_reward = delta_psi * self._yaw_reward_weight
            else:
                delta_psi = 0
                forward_reward = 0
            
            costs = ctrl_cost = self.control_cost(action, tendon_length_6)
            
            healthy_reward = 0
            
            terminated = self.terminated  
            if self._step_num > 1000:
                terminated = True
        
        else: #self._desired_action == "tracking":
            # ditch tracking reward
            ditch_rew_after = self._ditch_reward(xy_position_after)
            ditch_rew_before = self._ditch_reward(xy_position_before)
            forward_reward = ditch_rew_after - ditch_rew_before

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            healthy_reward = 0

            terminated = self.terminated  
            if self._step_num > 1000:
                terminated = True
        

        rewards = forward_reward + healthy_reward

        # if the contact between bars is too high, terminate the training run
        if np.any(self.data.cfrc_ext > 1500) or np.any(self.data.cfrc_ext < -1500):
            terminated = True

        #sum up the contact between each bar and multiply by the self._contact_with_self_penalty
        # contact_with_self_cost = 0
        # for j,contact in enumerate(self.data.contact):
        #     if contact.geom1 != 0 and contact.geom2 != 0: # neither geom is 0, which is ground. so contact is between bars
        #         forcetorque = np.zeros(6)
        #         mujoco.mj_contactForce(self.model, self.data, j, forcetorque)
        #         force_mag = np.sqrt(forcetorque[0]**2 + forcetorque[1]**2 + forcetorque[2]**2)
        #         contact_with_self_cost += self._contact_with_self_penalty* force_mag
        #         if force_mag > 1500 or force_mag < -1500:
        #           terminated = True


        
        observation, observation_with_noise = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            #"reward_contact_with_self": -contact_with_self_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "psi": psi_after,
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": self._x_velocity,
            "y_velocity": self._y_velocity,
            "tendon_length": tendon_length,
            "real_observation": observation,
            "forward_reward": forward_reward,
            "waypt": self._waypt,
            "oripoint": self._oripoint,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs #- contact_with_self_cost

        self._step_num += 1

        if self.render_mode == "human":
            self.render()
        if self._use_obs_noise == False:
            return observation, reward, terminated, False, info
        else:
            return observation_with_noise, reward, terminated, False, info

    def _get_obs(self):
        
        
        """ rotation_r01 = Rotation.from_matrix(self.data.geom("r01").xmat.reshape(3,3)).as_quat() # 4
        rotation_r23 = Rotation.from_matrix(self.data.geom("r23").xmat.reshape(3,3)).as_quat() # 4
        rotation_r45 = Rotation.from_matrix(self.data.geom("r45").xmat.reshape(3,3)).as_quat() # 4 """

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()

        pos_center = (pos_r01_left_end + pos_r01_right_end + pos_r23_left_end + pos_r23_right_end + pos_r45_left_end + pos_r45_right_end) / 6

        pos_rel_s0 = pos_r01_left_end - pos_center # 3
        pos_rel_s1 = pos_r01_right_end - pos_center # 3
        pos_rel_s2 = pos_r23_left_end - pos_center # 3
        pos_rel_s3 = pos_r23_right_end - pos_center # 3
        pos_rel_s4 = pos_r45_left_end - pos_center # 3
        pos_rel_s5 = pos_r45_right_end - pos_center # 3

        rng = np.random.default_rng()
        random = rng.standard_normal(size=3)
        pos_rel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s0 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s1 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s2 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s3 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s4 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s5 # 3

        # do not include positional data in the observation
        # position_r01 = self.data.geom("r01").xvelp
        # position_r23 = self.data.geom("r23").xvelp
        # position_r45 = self.data.geom("r45").xvelp


        tendon_lengths = self.data.ten_length # 9
        
        random = rng.standard_normal(size=9)
        tendon_lengths_with_noise = random * self._obs_noise_tendon_stdev + tendon_lengths # 9

        observation = np.concatenate((pos_rel_s0,pos_rel_s1,pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5,\
                                    tendon_lengths))
        observation_with_noise = np.concatenate((pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise,\
                                    tendon_lengths_with_noise))
        
        if self._use_cap_velocity:
            velocity = self.data.qvel # 18

            vel_lin_r01 = np.array([velocity[0], velocity[1], velocity[2]])
            vel_ang_r01 = np.array([velocity[3], velocity[4], velocity[5]])
            vel_lin_r23 = np.array([velocity[6], velocity[7], velocity[8]])
            vel_ang_r23 = np.array([velocity[9], velocity[10], velocity[11]])
            vel_lin_r45 = np.array([velocity[12], velocity[13], velocity[14]])
            vel_ang_r45 = np.array([velocity[15], velocity[16], velocity[17]])

            s0_r01_pos = pos_r01_left_end - self.data.body("r01_body").xpos.copy()
            s1_r01_pos = pos_r01_right_end - self.data.body("r01_body").xpos.copy()
            s2_r23_pos = pos_r23_left_end - self.data.body("r23_body").xpos.copy()
            s3_r23_pos = pos_r23_right_end - self.data.body("r23_body").xpos.copy()
            s4_r45_pos = pos_r45_left_end - self.data.body("r45_body").xpos.copy()
            s5_r45_pos = pos_r45_right_end - self.data.body("r45_body").xpos.copy()

            vel_s0 = vel_lin_r01 + np.cross(vel_ang_r01, s0_r01_pos) # 3
            vel_s1 = vel_lin_r01 + np.cross(vel_ang_r01, s1_r01_pos) # 3
            vel_s2 = vel_lin_r23 + np.cross(vel_ang_r23, s2_r23_pos) # 3
            vel_s3 = vel_lin_r23 + np.cross(vel_ang_r23, s3_r23_pos) # 3
            vel_s4 = vel_lin_r45 + np.cross(vel_ang_r45, s4_r45_pos) # 3
            vel_s5 = vel_lin_r45 + np.cross(vel_ang_r45, s5_r45_pos) # 3

            random = rng.standard_normal(size=3)
            vel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s0 # 3
            random = rng.standard_normal(size=3)
            vel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s1 # 3
            random = rng.standard_normal(size=3)
            vel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s2 # 3
            random = rng.standard_normal(size=3)
            vel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s3 # 3
            random = rng.standard_normal(size=3)
            vel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s4 # 3
            random = rng.standard_normal(size=3)
            vel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s5 # 3

            observation = np.concatenate((pos_rel_s0,pos_rel_s1,pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5,\
                                        vel_s0, vel_s1, vel_s2, vel_s3, vel_s4, vel_s5,\
                                        tendon_lengths))
            observation_with_noise = np.concatenate((pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise,\
                                        vel_s0_with_noise, vel_s1_with_noise, vel_s2_with_noise, vel_s3_with_noise, vel_s4_with_noise, vel_s5_with_noise,\
                                        tendon_lengths_with_noise))

        if self._desired_action == "tracking" or self._desired_action == "aiming":
            tracking_vec = self._waypt - pos_center[:2]
            tgt_drct = tracking_vec / np.linalg.norm(tracking_vec)
            pos_center_noise_del = (pos_rel_s0_with_noise + pos_rel_s1_with_noise + pos_rel_s2_with_noise + pos_rel_s3_with_noise + pos_rel_s4_with_noise + pos_rel_s5_with_noise)/6
            tracking_vec_with_noise = tracking_vec - pos_center_noise_del[:2]
            tgt_drct_with_noise = tracking_vec_with_noise / np.linalg.norm(tracking_vec_with_noise)

            tgt_yaw = np.array([np.arctan2(tgt_drct[1], tgt_drct[0])])
            tgt_yaw_with_noise = np.array([np.arctan2(tgt_drct_with_noise[1], tgt_drct_with_noise[0])])

            observation = np.concatenate((observation,\
                                          tracking_vec, tgt_yaw))
            observation_with_noise = np.concatenate((observation_with_noise,\
                                                     tracking_vec_with_noise, tgt_yaw_with_noise))

        return observation, observation_with_noise

    def _angle_normalize(self, theta):
        if theta > np.pi:
            return self._angle_normalize(theta - 2 * np.pi)
        elif theta <= -np.pi:
            return self._angle_normalize(theta + 2 * np.pi)
        else:
            return theta
    
    def _ditch_reward(self, xy_position):
        pointing_vec = self._waypt - self._oripoint
        dist_pointing = np.linalg.norm(pointing_vec)
        pointing_vec_norm = pointing_vec / dist_pointing

        tracking_vec = self._waypt - xy_position
        dist_along = np.dot(tracking_vec, pointing_vec_norm)
        dist_bias = np.linalg.norm(tracking_vec - dist_along*pointing_vec_norm) * np.sign(np.linalg.det(np.array([tracking_vec, pointing_vec_norm])))

        waypt_yaw = self._angle_normalize(np.arctan2(pointing_vec[1], pointing_vec[0]) - self._reset_psi)
        center_bias = (dist_along/dist_pointing - 1.0) * dist_pointing * np.tan(waypt_yaw)

        ditch_rew = self._ditch_reward_max * (1.0 - np.abs(dist_along)/dist_pointing) * np.exp(-(dist_bias-center_bias)**2 / (2*self._ditch_reward_stdev**2))
        waypt_rew = self._waypt_reward_amplitude * np.exp(-np.linalg.norm(xy_position - self._waypt)**2 / (2*self._waypt_reward_stdev**2))
        return ditch_rew+waypt_rew

    def _reset_cap_size(self, noise_range):
        cap_size_noise_low, cap_size_noise_high = noise_range
        cap_size = np.random.uniform(low=cap_size_noise_low, high=cap_size_noise_high)

        for i in range(self.model.ngeom):
            geom_name = self.model.geom_names[i].decode('utf-8')
            print(f"Index: {i}, Name: {geom_name}")

        cap_0_id = self.model.geom_name2id('s0')
        cap_1_id = self.model.geom_name2id('s1')
        cap_2_id = self.model.geom_name2id('s2')
        cap_3_id = self.model.geom_name2id('s3')
        cap_4_id = self.model.geom_name2id('s4')
        cap_5_id = self.model.geom_name2id('s5')

        self.model.geom_size[cap_0_id] = cap_size
        self.model.geom_size[cap_1_id] = cap_size
        self.model.geom_size[cap_2_id] = cap_size
        self.model.geom_size[cap_3_id] = cap_size
        self.model.geom_size[cap_4_id] = cap_size
        self.model.geom_size[cap_5_id] = cap_size
        return


    def reset_model(self):
        self._psi_wrap_around_count = 0

        if self._use_cap_size_noise == True:
            self._reset_cap_size(self._cap_size_noise_range)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        if self._desired_action == "turn" or self._desired_action == "tracking" or self._desired_action == "aiming":
            self.set_state(qpos, qvel)
        
        position_r01 = qpos[0:3]
        rotation_r01 = Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler('xyz')
        position_r23 = qpos[7:10]
        rotation_r23 = Rotation.from_quat([qpos[11], qpos[12], qpos[13], qpos[10]]).as_euler('xyz')
        position_r45 = qpos[14:17]
        rotation_r45 = Rotation.from_quat([qpos[18], qpos[19], qpos[20], qpos[17]]).as_euler('xyz')

        ux = 0
        uy = 0
        uz = 1
        theta = np.random.uniform(low=self._min_reset_heading, high=self._max_reset_heading)
        R = np.array([[np.cos(theta)+ux**2*(1-np.cos(theta)), 
                       ux*uy*(1-np.cos(theta))-uz*np.sin(theta),
                       ux*uz*(1-np.cos(theta))+uy*np.sin(theta)],
                       [uy*ux*(1-np.cos(theta))+uz*np.sin(theta),
                        np.cos(theta)+uy**2*(1-np.cos(theta)),
                        uy*uz*(1-np.cos(theta)-ux*np.sin(theta))],
                        [uz*ux*(1-np.cos(theta)) -uy*np.sin(theta),
                         uz*uy*(1-np.cos(theta)) + ux*np.sin(theta),
                         np.cos(theta)+uz**2*(1-np.cos(theta))]])

        
        position_r01_new = (R @ position_r01.reshape(-1,1)).squeeze()
        position_r23_new = (R @ position_r23.reshape(-1,1)).squeeze()
        position_r45_new = (R @ position_r45.reshape(-1,1)).squeeze()
        rot_quat_r01_new = Rotation.from_euler('xyz', rotation_r01 + [0, 0, theta]).as_quat()
        rot_quat_r01_new = [rot_quat_r01_new[3], rot_quat_r01_new[0], rot_quat_r01_new[1], rot_quat_r01_new[2]]
        rot_quat_r23_new = Rotation.from_euler('xyz', rotation_r23 + [0, 0, theta]).as_quat()
        rot_quat_r23_new = [rot_quat_r23_new[3], rot_quat_r23_new[0], rot_quat_r23_new[1], rot_quat_r23_new[2]]
        rot_quat_r45_new = Rotation.from_euler('xyz', rotation_r45 + [0, 0, theta]).as_quat()
        rot_quat_r45_new = [rot_quat_r45_new[3], rot_quat_r45_new[0], rot_quat_r45_new[1], rot_quat_r45_new[2]]

        qpos_new = np.concatenate((position_r01_new, rot_quat_r01_new, position_r23_new, rot_quat_r23_new,
                                   position_r45_new, rot_quat_r45_new))
        self.set_state(qpos_new, qvel)

        rng = np.random.default_rng()
        random = rng.standard_normal(size=6)
        tendons = random*self._tendon_reset_stdev + self._tendon_reset_mean
        for i in range(tendons.size):
            if tendons[i] > self._tendon_max_length:
                tendons[i] = self._tendon_max_length
            elif tendons[i] < self._tendon_min_length:
                tendons[i] = self._tendon_min_length
        
        for i in range(50):
            self.do_simulation(tendons, self.frame_skip)


        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_before = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_before = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3
        orientation_vector_before = left_COM_before - right_COM_before
        self._reset_psi = np.arctan2(-orientation_vector_before[0], orientation_vector_before[1])
        

        if self._desired_action == "tracking":
            self._oripoint = np.array([(left_COM_before[0]+right_COM_before[0])/2, (left_COM_before[1]+right_COM_before[1])/2])
            min_waypt_range, max_waypt_range = self._waypt_range
            min_waypt_angle, max_waypt_angle = self._waypt_angle_range
            waypt_length = np.random.uniform(min_waypt_range, max_waypt_range)
            waypt_yaw = np.random.uniform(min_waypt_angle, max_waypt_angle) + self._reset_psi
            if self._is_test == True:
                kmm_length = 0.5
                kmm_yaw = 0.5
                waypt_length = kmm_length*max_waypt_range + (1-kmm_length)*min_waypt_range
                waypt_yaw = (kmm_yaw*max_waypt_angle + (1-kmm_yaw)*min_waypt_angle) + self._reset_psi
            self._waypt = np.array([self._oripoint[0] + waypt_length * np.cos(waypt_yaw), self._oripoint[1] + waypt_length * np.sin(waypt_yaw)])
            if self._is_test == True: # for test3
                self._waypt = np.array([0, 0]) # for test3
        elif self._desired_action == "aiming":
            self._oripoint = np.array([(left_COM_before[0]+right_COM_before[0]/2), (left_COM_before[1]+right_COM_before[1])/2])
            min_waypt_range, max_waypt_range = self._waypt_range
            min_waypt_angle = -np.pi
            max_waypt_angle = np.pi
            waypt_length = np.random.uniform(min_waypt_range, max_waypt_range)
            waypt_yaw = np.random.uniform(min_waypt_angle, max_waypt_angle) + self._reset_psi
            if self._is_test == True:
                kmm_length = 0.5
                kmm_yaw = 0.75
                waypt_length = kmm_length*max_waypt_range + (1-kmm_length)*min_waypt_range
                waypt_yaw = (kmm_yaw*max_waypt_angle + (1-kmm_yaw)*min_waypt_angle) + self._reset_psi
            self._waypt = np.array([self._oripoint[0] + waypt_length * np.cos(waypt_yaw), self._oripoint[1] + waypt_length * np.sin(waypt_yaw)])
            if self._is_test == True: # for test3
                self._waypt = np.array([0, 0]) # for test3
                
        self._step_num = 0
        for i in range(self._reward_delay_steps):
            self.step(tendons)
        observation, observation_with_noise = self._get_obs()

        if self._use_obs_noise == False:
            return observation
        else:
            return observation_with_noise

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
