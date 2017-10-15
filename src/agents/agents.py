import numpy as np
from util.schedules.linear_schedule import LinearSchedule


class DQNAgent(object):
    def __init__(self, num_actions, q_func, schedule_time_steps, initial_time_step, final_p):
        """
        DQNの学習に用いるエージェント．
        ε-greedyによって行動を選択する．
        εは指定のステップから線形現象する．

        Parameters
        ----------
        num_actions: int
            環境の行動数
        q_func: model.CNN
            Q関数
        schedule_time_steps: int
            ε-greedyにおけるεを減少させるのに有するステップ数
        initial_time_step: int
            εを減少させ始めるステップ数
        final_p: float
            εの最終値
        """
        self.num_actions = num_actions
        self.q_func = q_func
        self.epsilon = LinearSchedule(schedule_time_steps,
                                      initial_time_step,
                                      initial_p=1.0,
                                      final_p=final_p)

    def action(self, t, obs):
        """
        ε-greedyに従って行動を選択．

        Parameters
        ----------
        t: int
            時間ステップ
        obs: np.ndarray
            Q-Networkへの入力

        Returns
        ----------
        action: int
            選択した行動
        """
        if self.epsilon.value(t) >= np.random.rand():
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_func.q_values.eval(
                feed_dict={self.q_func.s: [obs]}))
        return action


class DQNTestAgent(object):
    def __init__(self, num_actions, q_func):
        """
        DQNのテスト実行に用いるエージェント．
        DQNAgentにおいて，
        ε=0.0と固定したものに相当．

        Parameters
        ----------
        num_actions: int
            環境の行動数
        q_func: model.CNN
            Q関数
        """
        self.num_actions = num_actions
        self.q_func = q_func

    def action(self, t, obs):
        """
        greedyに行動を選択．

        Parameters
        ----------
        t: int
            時間ステップ
        obs: np.ndarray
            Q-Networkへの入力

        Returns
        ----------
        action: int
            選択した行動
        """
        action = np.argmax(self.q_func.q_values.eval(
            feed_dict={self.q_func.s: [obs]}))
        return action
