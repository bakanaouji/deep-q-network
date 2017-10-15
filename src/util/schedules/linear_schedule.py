class LinearSchedule(object):
    def __init__(self, schedule_time_steps, initial_time_step, final_p, initial_p=1.0):
        self.schedule_time_steps = schedule_time_steps
        self.initial_time_step = initial_time_step
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        t = max(0, t - self.initial_time_step)
        fraction = min(float(t) / self.schedule_time_steps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
