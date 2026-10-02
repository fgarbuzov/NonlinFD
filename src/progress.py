from time import time
from datetime import datetime, timedelta

__all__ = ['ProgressBarText']


class ProgressBarText():
    def __init__(self, values, length=30):
        self.values = values
        self.idx = 0
        self.bar_idx = 0
        self.max_idx = len(values)
        self.start_time = 0
        self.elapsed = 'unknown'
        self.estimated = 'unknown'
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == 0:
            self.start_time = time()
        else:
            cur_time = time()
            el = cur_time - self.start_time
            est =  self.max_idx/self.idx * el
            self.elapsed = timedelta(seconds=round(el))
            self.estimated = timedelta(seconds=round(est))
        printProgressBar(self.idx, self.max_idx, length=self.length, 
                         prefix='Progress:', 
                         suffix='Elapsed: {}, Estimated: {}'.format(
                             self.elapsed, self.estimated))
        if self.idx == self.max_idx:
            raise StopIteration
        value = self.values[self.idx]
        self.idx += 1
        return value

# Print iterations progress, 
# source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, 
                     length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration - Required : current iteration (Int)
        total     - Required : total iterations (Int)
        prefix    - Optional : prefix string (Str)
        suffix    - Optional : suffix string (Str)
        decimals  - Optional : positive number of decimals in percent complete (Int)
        length    - Optional : character length of bar (Int)
        fill      - Optional : bar fill character (Str)
        printEnd  - Optional : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()