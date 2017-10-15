class LinearSchedule(object):
    def __init__(self, schedule_time_steps, initial_time_step, final_p, initial_p=1.0):
        """
        線形に推移する値を管理するクラス．
        指定の時間で初期値から推移を開始する．
        指定の時間をかけて最終値まで線形に推移．

        Parameters
        ----------
        schedule_time_steps: int
            最終値まで推移するのに要する時間
        initial_time_step: int
            推移を始める時間
        final_p: float
            最終値
        initial_p: float
            初期値
        """
        self.schedule_time_steps = schedule_time_steps
        self.initial_time_step = initial_time_step
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """
        指定時間の値を取得．

        Parameters
        ----------
        t: int
            取得したい値の時間
        """
        t = max(0, t - self.initial_time_step)
        fraction = min(float(t) / self.schedule_time_steps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
