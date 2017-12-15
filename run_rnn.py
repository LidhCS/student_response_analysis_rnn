#!/usr/bin/python
# -*- coding:utf-8 -*
import logging
import numpy as np
import pandas as pd
import itertools as its
from scipy import sparse as sp
from collections import namedtuple
from data_preprocess.wrapper import load_data, DataOpts
from data_preprocess.constants import USER_IDX_KEY
from data_preprocess.rnn import build_nn_data
from data_preprocess.splitting_utils import split_data
from keras_rnn import rnn_keras
from config import source,data_file,num_folds,remove_skill_nans,seed,drop_duplicates,\
		     item_id_col,use_correct,use_hints,proportion_students_retained

# logging.basicConfig(filename='logger.log', level=logging.INFO)
logging.basicConfig(level=logging.INFO)

data_opts = DataOpts(item_id_col=item_id_col,
                     use_correct=use_correct,
                     remove_skill_nans=remove_skill_nans,
                     seed = seed,
                     use_hints=use_hints,
                     drop_duplicates=drop_duplicates,
                     proportion_students_retained=proportion_students_retained)
data, _, item_ids= load_data(data_file, source, data_opts)
num_questions = len(item_ids)
data_folds = split_data(data,num_folds=num_folds,seed = seed)

rnn_keras(data_folds,num_questions,data_opts.seed,data_opts.use_correct,data_opts.use_hints)


