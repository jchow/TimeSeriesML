import logging


class LoggerMixin(object):
    def __init__(self, file='temp.log', loglevel=logging.INFO):
        self.file = file
        self.loglevel = loglevel

    @property
    def logger(self):
        name = '.'.join([
            self.__module__,
            self.__class__.__name__
        ])

        logger = logging.getLogger(name)
        fh = logging.FileHandler(file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(self.loglevel)

        return logger