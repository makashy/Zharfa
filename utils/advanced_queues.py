""" Advanced Queue, a subclass of Queue
"""
# import time
import multiprocessing
from multiprocessing.queues import Empty, Full, JoinableQueue


class AdvancedQueue(JoinableQueue):
    """Advanced Queue
    This queue has additional methodes relative to queues.JoinableQueue.
    """
    def __init__(self, maxsize=-1):
        super().__init__(maxsize, ctx=multiprocessing.get_context())

    def empty_out(self):
        """Empties the queue
        """
        while True:
            try:
                self.get_nowait()
            except Empty:
                break

    def empty_and_put(self, item):
        """Empties the queue then enqueues an item
        Arguments:
            item: item to be enqueued
        """
        self.empty_out()
        # time.sleep(0.01) # TODO: why should here be a big delay or a exception handling
        # self.put_nowait(item)
        while True:
            try:
                self.put_nowait(item)
                break
            except Full:
                self.empty_out()

    def empty_and_get(self):
        """Dequeues the newest item and empties the queue
        Returns:
            usable: whether the item is usable or not
            item: dequeued item
        """
        usable = False
        item = False
        while True:
            try:
                item = self.get_nowait()
                usable = True
            except Empty:
                break
        return usable, item
