import sys

class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# Usage:
# sys.stdout = Logger("aa.log", sys.stdout)
# sys.stderr = Logger("aa.log", sys.stderr)       # redirect std err, if necessary