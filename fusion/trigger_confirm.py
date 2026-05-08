import time

class TriggerConfirm:
    def __init__(self, trigger_thresh=60, confirm_duration=10):
        self.trigger_thresh = trigger_thresh
        self.confirm_duration = confirm_duration
        self.trigger_time = None
        self.fatigue_confirmed = False

    def update(self, cfi):
        now = time.time()

        if cfi is None:
            self.trigger_time = None
            self.fatigue_confirmed = False
            return False

        if cfi >= self.trigger_thresh:
            if self.trigger_time is None:
                self.trigger_time = now
            elif (now - self.trigger_time) >= self.confirm_duration:
                self.fatigue_confirmed = True
        else:
            self.trigger_time = None
            self.fatigue_confirmed = False

        return self.fatigue_confirmed

    def time_remaining(self):
        if self.trigger_time is None:
            return None
        elapsed = time.time() - self.trigger_time
        return max(0.0, self.confirm_duration - elapsed)
