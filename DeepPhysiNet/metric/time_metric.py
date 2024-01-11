'''
@anthor: Wenyuan Li
@desc: Time Metric
@date: 2020/5/17
'''
import datetime

class TimeMetric(object):

    def __init__(self):
        self.start_time=datetime.datetime.now()

    def start(self):
        self.start_time=datetime.datetime.now()
    def reset(self):
        self.start_time=datetime.datetime.now()


    def get_time_ms(self):
        self.end_time=datetime.datetime.now()
        ms_time=(self.end_time-self.start_time).seconds*1000+\
                (self.end_time-self.start_time).microseconds/1000.
        return ms_time
    def get_time(self):
        self.end_time = datetime.datetime.now()
        return (self.end_time-self.start_time).seconds

    def get_fps(self,toal_frames):
        total_time=self.get_time()+1e-6
        fps=toal_frames/total_time
        return fps