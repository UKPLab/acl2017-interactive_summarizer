from __future__ import print_function
class FlightRecorder(object):
    """
    Keeps a list of records, plus the union of all of them, too.
    """

    def __init__(self):
        self.recordings = []

        self.total=Record(set(), set(), set())

    def record(self, accept=None, reject=None, implicit_reject=None):
        accept = set(accept)
        reject = set(reject)
        implicit_reject = set(implicit_reject)
        self.recordings.append(Record(accept, reject, implicit_reject))

        if accept is not None:
            self.total.accept |= accept
            # self.accepted_concepts=accept
            # self.total_accept += accept
        if reject is not None:
            self.total.reject |= reject
            # self.rejected_concepts=reject
            # self.total_reject += reject
        if implicit_reject is not None:
            self.total.implicit_reject |= implicit_reject
            # self.implicit_reject = implicit_reject
            # self.total_implicit_reject += implicit_reject

    def latest(self):
        """
            Returns the last added Record, or an empty Record
        :return:
        """
        if len(self.recordings) == 0:
            return Record(set(), set(), set())
        return self.recordings[-1:][0]

    def union(self):
        return self.total

    def clear(self):
        self.__init__()

class Record(object):
    def __init__(self, accept=set(), reject=set(), implicit_reject=set()):
        """
        @type accept: set[str]
        @type reject: set[str]
        @type implicit_reject: set[str]
        """
        self.accept=accept
        self.reject=reject
        self.implicit_reject=implicit_reject