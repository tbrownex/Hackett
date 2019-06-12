# -*- coding: utf-8 -*-
"""
Abstract Dispatcher
Created on Thu Sep 25 10:28:52 2018

@author: will.cairns


This function will utilize a function list
ex: task_list = [filter, normalize, train, forecast]

Then it will iterate:
    For task in task_list:
        df = task(df)
"""
