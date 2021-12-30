import cv2
from gym_duckietown.envs import DuckietownEnv
import math 
import numpy as np
import json
def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False):
    environment = DuckietownEnv(
        domain_rand=domain_rand,
        max_steps=math.inf,
        map_name=map_name,
        randomize_maps_on_reset=False,
    )
    return environment

class InteractiveImitationLearning:
    """
    A class used to contain main imitation learning algorithm
    ...
    Methods
    -------
    train(samples, debug)
        start training imitation learning
    """

    def __init__(self, env, teacher, learner, horizon, episodes, test=False):
        """
        Parameters
        ----------
        env :
            duckietown environment
        teacher :
            expert used to train imitation learning
        learner :
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """

        self.environment = env
        self.teacher = teacher
        self.learner = learner
        self.test = test

        # from IIL
        self._horizon = horizon
        self._episodes = episodes
        self._horizon_factor = 1.04
        # data
        self._observations = []
        self._expert_actions = []

        # statistics
        self.learner_action = None
        self.learner_uncertainty = None

        self.teacher_action = None
        self.active_policy = True  # if teacher is active

        # internal count
        self._current_horizon = 0
        self._episode = 0

        # event listeners
        self._episode_done_listeners = []
        self._found_obstacle = False
        # steering angle gain
        self.gain = 10

        self.observation_num = 0

    def train(self, debug=False):
        """
        Parameters
        ----------
        teacher :
            expert used to train imitation learning
        learner :
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """
        self._debug = debug
        possible_maps = ['ETHZ_autolab_technical_track',
                        'LF-norm-loop',
                        'LF-norm-small_loop',
                        'LF-norm-techtrack',
                        'LF-norm-zigzag',
                        # '_custom_technical_floor',
                        # '_huge_C_floor',
                        # '_huge_V_floor',
                        # '_loop_duckiebots',
                        # '_loop_duckies',
                        # '_loop_dyn_duckiebots',
                        # '_myTestA',
                        # '_plus_floor',
                        # 'huge_loop',
                        # 'huge_loop2',
                        # 'multi_track',
                        # 'multi_track2'
                        ]
        for episode in range(self._episodes):
            print('-> Episode : {}'.format(episode))
            map_name = np.random.choice(possible_maps)
            print('-> Map Name : {}'.format(map_name))
            '''
            self.environment = launch_env(
                map_name,
                domain_rand=False,
                randomize_maps_on_reset=False,
            )
            '''
            self._episode = episode
            self._sampling()
            self._optimize()  # episodic learning
            self._on_episode_done()
            print("Self episode", self._episode)
            #self._horizon = int(self._horizon_factor * self._horizon)

        print("Number of learner actions:", self.learner_calls)
        print("Number of teacher actions", self.expert_calls)
        print("Number of additional expert calls", self.additional_expert_calls)
        print("Additional teacher portion", self.additional_expert_calls/(self.expert_calls + self.learner_calls))

        res = {}
        res["expert_actions"] = self.expert_calls
        res["learner_actions"] = self.learner_calls
        res["additional_expert_actions"] = self.additional_expert_calls
        res["add_expert_portion"] = self.additional_expert_calls/(self.expert_calls + self.learner_calls)

        with open("action_number.json", "w") as f:
            json.dump(res, f)


    def _sampling(self):
        observation = self.environment.render_obs()
        for horizon in range(self._horizon):
            self._current_horizon = horizon
            action = self._act(observation)
            try:
                next_observation, reward, done, info = self.environment.step(
                    [action[0], action[1] * self.gain]
                )
                self.observation_num += 1
            except Exception as e:
                print(e)
            if self._debug:
                self.environment.render()
                #if self.test:
                #    cv2.imwrite('/content/drive/MyDrive/LF_Duckietown/out/{}_{}.png'.format(self._episode, horizon), observation)
            observation = next_observation

    # execute current control policy
    def _act(self, observation):
        if self._episode <= 1:  # initial policy equals expert's
            control_policy = self.teacher
            control_action = control_policy.predict(observation)
            self._aggregate(observation, control_action)

        else:
            control_policy = self._mix(observation)

        control_action = control_policy.predict(observation)

        self._query_expert(control_policy, control_action, observation)

        self.active_policy = control_policy == self.teacher
        if self.test:
            return self.learner_action

        return control_action

    def _query_expert(self, control_policy, control_action, observation):
        if control_policy == self.learner:
            self.learner_action = control_action    
        else:
            self.learner_action = self.learner.predict(observation)
        if not(self.test):
            var = self.learner.variance 

        if control_policy == self.teacher:
            self.teacher_action = control_action
        else:
            self.teacher_action = self.teacher.predict(observation)

        #if self.teacher_action is not None: # or var > 0.15: 
        #    self._aggregate(observation, self.teacher_action)

        if self.teacher_action[0] < 0.1:
            self._found_obstacle = True
        else:
            self._found_obstacle = False

    def _mix(self):
        raise NotImplementedError()

    def _aggregate(self, observation, action):


        if not (self.test):
            self._observations.append(observation)
            self._expert_actions.append(action)

    def _optimize(self):
        if not (self.test):
            self.learner.optimize(self._observations, self._expert_actions, self._episode)
            print("saving model")
            self.learner.save()

    # TRAINING EVENTS

    # triggered after an episode of learning is done
    def on_episode_done(self, listener):
        self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        for listener in self._episode_done_listeners:
            listener.episode_done(self._episode)
        self.environment.reset()

