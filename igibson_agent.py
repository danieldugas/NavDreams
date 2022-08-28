import numpy as np
import os

ACTION_DIM = 2
LINEAR_VEL_DIM = 0
ANGULAR_VEL_DIM = 1

def make_dir_if_not_exists(dir_):
    try:
        os.makedirs(dir_)
    except OSError:
        if not os.path.isdir(dir_):
            raise

class RandomAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, observations):
        action = np.random.uniform(low=-1, high=1, size=(ACTION_DIM,))
        return action

class RandomAgentAlsoRec:
    def __init__(self):
        # params
        self.archive_dir=os.path.expanduser("~/navdreams/datasets/V/igibsonchallenge")
        # vars
        self.scans = []
        self.robotstates = []
        self.actions = []
        self.rewards = []
        self.dones = []
        pass

    def reset(self):
        if len(self.dones) != 0:
            self.dones[-1] = 1
        pass

    def act(self, observations):
        action = np.random.uniform(low=-1, high=1, size=(ACTION_DIM,))
        self.scans.append(observations["rgb"])
        self.robotstates.append(observations["sensor"])
        self.actions.append(action)
        self.rewards.append(0)
        self.dones.append(0)

        if len(self.scans) >= 1000:
            self.dones[-1] = 1
            scans = np.array(self.scans)
            robotstates = np.array(self.robotstates)
            actions = np.array(self.actions)
            rewards = np.array(self.rewards)
            dones = np.array(self.dones)
            data = dict(scans=scans, robotstates=robotstates, actions=actions, rewards=rewards, dones=dones)
            if self.archive_dir is not None:
                make_dir_if_not_exists(self.archive_dir)
                archive_path = os.path.join(
                    self.archive_dir, "{:03}_scans_robotstates_actions_rewards_dones.npz".format(n)
                    )
                np.savez_compressed(archive_path, **data)
                print(archive_path, "written.")
            self.scans = []
            self.robotstates = []
            self.actions = []
            self.rewards = []
            self.dones = []
        return action

class ForwardOnlyAgent(RandomAgent):
    def act(self, observations):
        action = np.zeros(ACTION_DIM)
        action[LINEAR_VEL_DIM] = 1.0
        action[ANGULAR_VEL_DIM] = 0.0
        return action


if __name__ == "__main__":
    obs = {
        'depth': np.ones((180, 320, 1)),
        'rgb': np.ones((180, 320, 3)),
        'sensor': np.ones((2,))
    }

    agent = RandomAgent()
    action = agent.act(obs)
    print('action', action)

    agent = ForwardOnlyAgent()
    action = agent.act(obs)
    print('action', action)

