import os
import configparser

PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(PATH, "config.ini"))
