import numpy as np
import os
from PIL import Image
import json 
RISK_THRESHOLD = 0.5

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
        self.observation_save_path = "/home/plparent/challenge-aido_LF-baseline-dagger-pytorch/learning/results"
        self.observation_num = 0

        # from IIL
        self._horizon = horizon
        self._episodes = episodes

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
        self.additional_expert_calls = 0

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
        for episode in range(self._episodes):
            self._episode = episode
            self._sampling()
            self._optimize()  # episodic learning
            self._on_episode_done()

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
            except Exception as e:
                print(e)
            if self._debug:
                self.environment.render()
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
            if abs(control_action[2]) >= RISK_THRESHOLD and not self.test:
                print("RISK", abs(control_action[2]))
                self.additional_expert_calls += 1
                control_policy = self.teacher
                control_action = self.teacher.predict(observation)
                self._aggregate(observation, self.teacher_action)

        self._query_expert(control_policy, control_action, observation)

        self.active_policy = control_policy == self.teacher
        if self.test:
            return self.learner_action

        return control_action

    def _query_expert(self, control_policy, control_action, observation):
        '''
        
        if abs(control_action[2]) >= RISK_THRESHOLD and not self.test:
            control_policy = self.teacher
            control_action = self.teacher.predict(observation)
        '''
        

        if control_policy == self.learner:
            self.learner_action = control_action
        else:
            self.learner_action = self.learner.predict(observation)

        if control_policy == self.teacher:
            self.teacher_action = control_action
        else:
            self.teacher_action = self.teacher.predict(observation)

        #if self.teacher_action is not None:
        #    self._aggregate(observation, self.teacher_action)

        if self.teacher_action[0] < 0.1:
            self._found_obstacle = True
        else:
            self._found_obstacle = False

    def _mix(self):
        raise NotImplementedError()

    def _inject_noise(self, action):
        omega_noise = np.clip(action[1] + np.random.normal(), -np.pi/2, np.pi/2)
        return [action[0], omega_noise, omega_noise - action[1]]

    def _aggregate(self, observation, action):
        if not (self.test):
            self._observations.append(observation)
            self._expert_actions.append(action)
            self._observations.append(observation)
            self._expert_actions.append(self._inject_noise(action))
        else:            
            img = Image.fromarray(observation, 'RGB')
            #name = 'observation' + str(self.observation_num) + policy_str + 'learner.png'
            
            #img.save(os.path.join(self.observation_save_path, name))
            self.observation_num += 1

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
