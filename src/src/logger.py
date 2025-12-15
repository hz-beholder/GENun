import os
import pickle
import random
import logging
import types

class Logger(object):
    def __init__(self, index=None, path='logs/', always_save=True):
        if index is None:
            index = '{:06x}'.format(random.getrandbits(6 * 4))
        self.index = index
        self.filename = os.path.join(path, '{}.p'.format(self.index))
        self._dict = {}
        self.logs = []
        self.always_save = always_save

    def __getitem__(self, k):
        return self._dict[k]

    def __setitem__(self,k,v):
        self._dict[k] = v

    @staticmethod
    def load(filename, path='logs/'):
        if not os.path.isfile(filename):
            filename = os.path.join(path, '{}.p'.format(filename))
        if not os.path.isfile(filename):
            raise ValueError("{} is not a valid filename".format(filename))
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def save(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self, f)

    def get(self, _type):
        l = [x for x in self.logs if x['_type'] == _type]
        l = [x['_data'] if '_data' in x else x for x in l]
        # if len(l) == 1:
        #     return l[0]
        return l

    def append(self, _type, *args, **kwargs):
        kwargs['_type'] = _type
        if len(args)==1:
            kwargs['_data'] = args[0]
        elif len(args) > 1:
            kwargs['_data'] = args
        self.logs.append(kwargs)
        if self.always_save:
            self.save()
            
def log_newline(self, how_many_lines=1):
    # Switch formatter, output a blank line
    self.handler.setFormatter(self.blank_formatter)

    for i in range(how_many_lines):
        self.info('')

    # Switch back
    self.handler.setFormatter(self.formatter)
    
# '''Shash: Changed to avoid duplicate logging - https://stackoverflow.com/a/7175288'''
loggers = {}
def create_logger(folder, logname):
    global loggers
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    if loggers.get(f'{folder}-{logname}'):
        return loggers.get(f'{folder}-{logname}')
    else:
        logger = logging.getLogger(f'{folder}-{logname}')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(message)s")
        # formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
        
        # file logger
        if not os.path.isdir(folder):
            os.mkdir(folder)
        logfn = os.path.join(folder, f'{logname}.log')
        new_logger = os.path.exists(logfn)
        
        fh = logging.FileHandler(logfn, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.handler = fh
        logger.formatter = formatter
        
        ## add a blank formatter and new empty line in the logger
        blank_formatter = logging.Formatter(fmt="")
        logger.blank_formatter = blank_formatter
        logger.newline = types.MethodType(log_newline, logger)        
        if new_logger:
            logger.newline()
        
        # console logg
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        loggers[f'{folder}-{logname}'] = logger
    return logger