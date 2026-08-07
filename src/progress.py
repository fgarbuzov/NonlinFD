from time import time
from datetime import datetime, timedelta

__all__ = ['Progress', 'Status', 'ProgressBarText']

class Status():
    def __init__(self, log=None, show=True):
        if show:
            from ipywidgets import HTML
            from IPython.display import display
            self.widget = HTML()       
            display(self.widget)
        else:
            self.widget = None
        self.log = log
        
    def show(self, s, tooltip=None):
        if self.widget is not None:
            if tooltip is not None:
                title = 'title="%s"' % tooltip
            else:
                title = ''
            self.widget.value = ('<div class="output_area" %s>'
                                 '<pre>%s</pre>'
                                 '</div>' % (title, s))
                             
        if self.log is not None:
            with open(self.log, 'w') as f:
                print(s, file=f)
                if tooltip is not None:
                    print(tooltip, file=f)
                    
    def __del__(self):
        if self.widget is not None:
            self.widget.close()        
    

class Progress(Status):
    def __init__(self, iterable_or_total=None, total=None, show_res=False, 
                 update_interval=1.0, log=None, show=True):
        try:
            # if any iterable given - use it
            self.iterable = iter(iterable_or_total)
            if total is None:
                try:
                    # if has len - it's the total
                    self.total = len(iterable_or_total)
                except TypeError:
                    # iterable without len - indeterminate
                    self.total = None
            else:
                self.total = total
        except TypeError:
            # not iterable - assume as total
            self.total = iterable_or_total
            self.iterable = iter(range(iterable_or_total))
        
        self.char = '*'
        self.width = 36
        self.show_res = show_res
        self.update_interval = update_interval

        self.start_time = time()
        self.update_time = None
        self.n = 0
        self.res = ''
        super().__init__(log=log, show=show)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.update(self.res)
        self.res = next(self.iterable)
        return self.res

    def update(self, res=''):
        if self.update_time is not None:
            if time() - self.update_time < self.update_interval:
                self.n += 1
                return
            
        full = self.width - 2 # width without brackets
        if self.total is not None:
            percent = '%d' % (self.n*100.0/self.total)
            fill = max(0, min(full, int(round(self.n*full/self.total))))
            prog_bar = '[' + self.char*fill + ' '*(full - fill) + ']'
            pct_place = len(prog_bar)//2 + 1 - len(percent)
            prog_bar = (prog_bar[:pct_place] + percent + '%' +
                        prog_bar[pct_place + len(percent) + 1:])
            done = '%d/%d' % (self.n, self.total) 
        else:
            fill = full//2
            end = self.n % (full + fill)
            start = max(min(end - fill, full), 0)
            end = max(min(end, full), 0)
            prog_bar = ('[' + ' '*start + self.char*(end - start) +
                        ' '*(full - end) + ']')
            done = str(self.n) 
            
        if self.show_res:
            prog_bar += ' %s (%s)' % (res, done)
        else:
            prog_bar += ' ' + done
        
        self.update_time = time()
        dt = self.update_time - self.start_time
        prog_bar += ', time: %s' % timedelta(seconds=round(dt))
        
        if (self.total is not None) and (0 < self.n < self.total):
            est = dt/self.n*self.total - dt
            prog_bar += ', est: %s' % timedelta(seconds=round(est))
            ETA = datetime.fromtimestamp(self.update_time + est)
            tooltip = 'ETA: %s' % ETA.replace(microsecond=0)
        else:
            tooltip = 'ETA is unknown'
            
        self.show(prog_bar, tooltip)
        self.n += 1




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