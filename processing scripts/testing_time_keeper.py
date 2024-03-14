#!/usr/bin/env python3
import rospy
import bisect


class TimeKeeper:
    """
    A class used to keep track of rospy.Time objects in sorted order.

    Attributes
    ----------
    times : list
        a list of tuples, where each tuple contains a time in nanoseconds and an index
    index : int
        the index for the next time to be added

    Methods
    -------
    update(time)
        Adds a new time to the list in a way that maintains the sorted order.
    """

    def __init__(self):
        """
        Initializes the TimeKeeper with an empty list of times and an index of 0.
        """
        self.times = []
        self.index = 0

    def update(self, time):
        """
        Adds a new time to the list in a way that maintains the sorted order.

        Parameters
        ----------
        time : rospy.Time
            the time to be added
        """
        # Convert rospy.Time to nanoseconds for sorting
        time_in_nsec = time.to_nsec()

        # Find the insertion point for the new time
        insertion_point = bisect.bisect(self.times, (time_in_nsec,))

        # Insert the new time at the correct position
        self.times.insert(insertion_point, (time_in_nsec, self.index))

        # Increment the index
        self.index += 1
