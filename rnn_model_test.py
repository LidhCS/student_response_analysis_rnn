#-*- coding:utf-8 -*-
import logging
import sys
import numpy as np
import pandas as pd
import itertools as its
from scipy import sparse as sp
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from test.wrapper import load_data, DataOpts
from data_preprocess.constants import USER_IDX_KEY
from data_preprocess.rnn import build_nn_data
from keras import models 
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras import backend as K
from keras.layers.core import Masking
from config import source,remove_skill_nans,seed,drop_duplicates,\
					item_id_col,use_correct,use_hints,proportion_students_retained

test_data_file = sys.argv[1] 

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

data_opts = DataOpts(item_id_col=item_id_col,
                     use_correct=use_correct,
                     remove_skill_nans=remove_skill_nans,
                     seed = seed,
                     use_hints=use_hints,
                     drop_duplicates=drop_duplicates,
                     proportion_students_retained=proportion_students_retained)
data, user_ids, item_ids= load_data(test_data_file, source, data_opts)
num_questions = 494
maxlength = 15847
student_ids = np.unique(data[USER_IDX_KEY])
test_idx = np.in1d(data[USER_IDX_KEY], student_ids)
test_idx_data = data[test_idx].copy()
test_data = build_nn_data(test_idx_data, num_questions,
							use_correct=data_opts.use_correct,
							use_hints=data_opts.use_hints)

interaction_length = []
data_xset = []
for userdata in test_data:
    userdata.history.append(userdata.next_answer[-1])
    data_xset.append(userdata.history)
    interaction_length.append(userdata.length)
with open('predict_result/student_history.txt','w') as f:
    for i in range(len(data_xset)):
        f.write(str(data_xset[i])+ '\n')
LOGGER.info('the number of students:{}'.format(len(data_xset)))
data_X = pad_sequences(data_xset,maxlen = maxlength,padding = 'post',value = -1)
#(sample,timestep,dimensions of data points)（样本数、时间步长、样本点本身的维度）
data_X = np.reshape(data_X,(data_X.shape[0],maxlength,1))
X = data_X / float(2*num_questions)

model = Sequential()
model.add(Masking(mask_value = -1, input_shape=(maxlength, 1)))
model.add(SimpleRNN(2*num_questions,activation='softmax'))
model.load_weights('model/rnn_keras_model_1.h5')
prediction = model.predict(X,verbose = 0)
prediction_sets = []
for index in range(len(data_xset)):
    predict_each = [(prediction[index][i] / (prediction[index][i-num_questions]+prediction[index][i])) for i in range(num_questions,2*num_questions,1)]
    count = 0
    student_question_number = []
    for k in range(len(predict_each)):
        if predict_each[k]>0.6:
            count +=1
            student_question_number.append(k)
    LOGGER.info(student_question_number)
    with open('predict_result/predict_result.txt','w') as f:
        for j in range((len(student_question_number))):
            f.write(str(student_question_number[j])+',')
    LOGGER.info('count:{}'.format(count))
    prediction_sets.append(predict_each)
with open('predict_result/prediction_prob.txt','w') as f:
    f.write(str(prediction_sets))

