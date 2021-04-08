# -*- coding:utf-8 -*-
# CREATED BY: bohuai jiang 
# CREATED ON: 2020/4/23 10:52 AM
# LAST MODIFIED ON:
# AIM: 定义model的标准
import sys
import time


def print_percent(current: int, max: int, header: str = ''):
    percent = float(current) / max * 100
    sys.stdout.write("\r{0}{1:.3g}%".format(header, percent))
    # sys.stdout.flush()


def print_time_cost(time_start: float, id: int, data_length: int, header:str =''):
    time_lapse = time.time() - time_start
    average_lapse = time_lapse / float((id + 1))
    time_remain = average_lapse * (data_length - (id + 1))
    print_percent(id, data_length,
                  header='{0} total_datasize {1} [{2}|{3}]， avg_time_pre_batch {4:0.4f}s, time_remain {5}s - '.format(
                      header,
                      data_length,
                      id,
                      data_length,
                      average_lapse,
                      time.strftime("%H:%M:%S", time.gmtime(time_remain)))
                  )
