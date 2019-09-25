"""Contains a class for measuring short duration of time
"""
import time

class Chronometer():
    """ A Chronometer
    """
    def __init__(self, counter=time.perf_counter, smoothing_factor=0.1):
        self.counter = counter
        self.start_time = counter()
        self.average = 0
        self.smoothing_factor = smoothing_factor

    def start(self):
        """Starts the chronometer
        """
        self.start_time = self.counter()

    def give_elapsed(self):
        """Calculates the elapsed time
        """
        return self.counter() - self.start_time

    def average_elapsed(self):
        """Calculates an average for a repetitive cycle
        """
        self.average = self.average + (self.give_elapsed() - self.average) * self.smoothing_factor
        return self.average
