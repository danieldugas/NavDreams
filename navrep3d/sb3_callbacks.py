import numpy as np
import time
from pandas import DataFrame
from navrep.tools.sb_eval_callback import save_log, print_statistics, save_model_if_improved

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class NavRep3DLogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Logs the performance of NavRep3DTrainEnv every N steps

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, logpath=None, savepath=None, eval_freq=10000, verbose=0):
        super(NavRep3DLogCallback, self).__init__(verbose)
        # self.model = None  # type: BaseRLModel
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        if "_ckpt" not in savepath:
            raise ValueError("expected checkpoint path to contain '_ckpt'")
        self.logpath = logpath
        self.savepath = savepath
        self.eval_freq = eval_freq
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # get episode_statistics
            env = self.training_env
            if isinstance(self.training_env, DummyVecEnv):
                S = DataFrame()
                for env in self.training_env.envs:
                    S = S.append(env.episode_statistics, ignore_index=True)
                S["total_steps"] *= len(self.training_env.envs)
                S = S.sort_values(by="total_steps", ignore_index=True)
            elif isinstance(self.training_env, SubprocVecEnv):
                S = DataFrame()
                episode_statistics_list = self.training_env.get_attr("episode_statistics")
                for episode_statistics in episode_statistics_list:
                    S = S.append(episode_statistics, ignore_index=True)
                S["total_steps"] *= self.training_env.num_envs
                S = S.sort_values(by="total_steps", ignore_index=True)
            else:
                S = env.episode_statistics

            new_S = S[self.last_len_statistics:]
            new_avg_reward = np.mean(new_S["reward"].values)

            save_log(S, self.logpath, self.verbose)
            print_statistics(new_S, 0, 0, self.n_calls, self.num_timesteps, self.verbose)
            save_model_if_improved(new_avg_reward, self.best_avg_reward, self.model,
                                   self.savepath.replace("_ckpt", "_bestckpt"))
            save_model_if_improved(1., [0], self.model, self.savepath)

            self.last_len_statistics = len(S)
        return True
