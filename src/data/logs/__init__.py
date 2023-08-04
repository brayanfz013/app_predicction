# coding=utf-8

import os 
LOGS_DIR = os.path.dirname(__file__)

FILE_LOGS = 'logger_p_data.log'
path_logs = os.path.join(LOGS_DIR,FILE_LOGS)

if not os.path.isfile(path_logs):
    open(path_logs, 'x',encoding='utf-8')
