import sys
from helpers import load_config
class Logger(object):
    def __init__(self, cfg_path):
        cfg_file = load_config(cfg_path)
        log_path = cfg_file["model"]["path"]+cfg_file["tracing"]["log_name"] +".out"
        print("log_path = ", log_path)
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
