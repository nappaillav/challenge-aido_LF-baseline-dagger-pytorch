import math
from .iil_learning import InteractiveImitationLearning
import numpy as np
from PIL import Image
import os
import cv2
class DAgger(InteractiveImitationLearning):
    """
    DAgger algorithm to mix policies between learner and expert
    Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." Proceedings of the fourteenth international conference on artificial intelligence and statistics. 2011.
    ...
    Methods
    -------
    _mix
        used to return a policy teacher / expert based on random choice and safety checks
    """

    def __init__(self, env, teacher, learner, horizon, episodes, save_observs,  save_observs_path, save_observs_autoencoder, alpha=0.5, test=False):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, test)
        # expert decay
        self.p = alpha
        self.alpha = self.p

        # thresholds used to give control back to learner once the teacher converges
        self.convergence_distance = 0.05
        self.convergence_angle = np.pi / 18

        # threshold on angle and distance from the lane when using the model to avoid going off track and env reset within an episode
        self.angle_limit = np.pi / 8
        self.distance_limit = 0.12


        self.save_observations = save_observs
        self.observation_num = 0
        self.observation_save_path = save_observs_path#"/home/dmhum/duckie/challenge-aido_LF-baseline-dagger-pytorch/learning/observations_test"

        self.observation_autoencoder_path = save_observs_autoencoder

        if not(os.path.exists(self.observation_save_path)):
            os.mkdir(self.observation_save_path)
        self.observation_teacher_path = os.path.join(self.observation_save_path, "teacher")
        self.observation_learner_path = os.path.join(self.observation_save_path, "learner")

        self.additional_expert_calls = 0

        self.expert_calls = 0
        self.learner_calls = 0
        self.obs_auto = 0

    def _mix(self, observation):
        observations = [observation]
        input_shape = (160, 120)

        loss_auto = self.learner.autoencoder_predict(observation).item()
        control_policy = np.random.choice(a=[self.teacher, self.learner], p=[self.alpha, 1.0 - self.alpha])
        if self._found_obstacle:
            return self.teacher
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except:
            return control_policy

        '''
        if self.active_policy:
            # keep using tecaher untill duckiebot converges back on track
            if not (abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad) < self.convergence_angle):
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give
            # control back to the expert
            if abs(lp.dist) > self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                return self.teacher
        '''
        if loss_auto > 0.02:
            observation = cv2.resize(observation, dsize=input_shape[::-1])
            #print("loss", loss_auto)
            img = Image.fromarray(observation, 'RGB')

            name = 'observation' + str(self.obs_auto) + "_" + str(loss_auto) + "_dist_" + str((lp.dist)) + '_.png'
            
            if self.save_observations:
                if not(os.path.exists(self.observation_autoencoder_path)):
                    os.mkdir(self.observation_autoencoder_path)          
                img.save(os.path.join(self.observation_autoencoder_path, name))
                self.obs_auto += 1
                print("loss", loss_auto)

            self.additional_expert_calls += 1
            self.expert_calls += 1
            teacher_action = self.teacher.predict(observation)
            self._aggregate(observation, teacher_action)
            return self.teacher


        if control_policy == self.learner:
            self.learner_calls += 1
            #print("LEARNER")
        elif control_policy == self.teacher:
            self.expert_calls += 1
            teacher_action = self.teacher.predict(observation)
            self._aggregate(observation, teacher_action)
            #print("EXPERT")
        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
