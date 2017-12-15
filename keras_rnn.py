#-*- coding:utf-8 -*-
import numpy as np
from data_preprocess.rnn import build_nn_data
from keras import backend as K
from keras.models import Sequential
from keras.layers import SimpleRNN,LSTM
from keras.preprocessing.sequence import pad_sequences
from keras import losses,optimizers
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.core import Masking
import tensorflow as tf
import logging

LOGGER = logging.getLogger(__name__)

def my_loss(y_true,y_pred):
	num = int(int(y_pred.shape[1]) / 2)
	row_list = [0] * (2*num)
	transform_list = [0] * (2*num)
	for i in range(num):
		row_list[i] = row_list[i+num] = 1
		transform_list[i] = row_list
		transform_list[i+num] = row_list
		row_list = [0] * (2*num)
	transform_matrix_1 = tf.convert_to_tensor(transform_list,dtype = 'float32')
	transform_matrix_2 = tf.matmul(y_true,transform_matrix_1)
	y_pred_valid = y_pred * transform_matrix_2
	y_pred_divide = tf.reduce_sum(y_pred_valid,axis=len(y_pred_valid.get_shape()) - 1, keep_dims=True)
	y_pred_new = y_pred_valid / y_pred_divide
	# y_pred_new = y_pred_valid_new * y_true
	return K.categorical_crossentropy(y_true,y_pred_new)

def my_accuracy(y_true,y_pred):
	num = int(int(y_pred.shape[1]) / 2)
	row_list = [0] * (2*num)
	transform_list = [0] * (2*num)
	for i in range(num):
		row_list[i] = row_list[i+num] = 1
		transform_list[i] = row_list
		transform_list[i+num] = row_list
		row_list = [0] * (2*num)
	transform_matrix_1 = tf.convert_to_tensor(transform_list,dtype = 'float32')
	transform_matrix_2 = tf.matmul(y_true,transform_matrix_1)
	y_pred_valid = y_pred * transform_matrix_2
	y_pred_divide = tf.reduce_sum(y_pred_valid,axis=len(y_pred_valid.get_shape()) - 1, keep_dims=True)
	y_pred_new = y_pred_valid / y_pred_divide
	return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred_new, axis=-1)), K.floatx()) 

def data_standardization(dataX,dataY,maxlength,num_questions):
	data_X = pad_sequences(dataX,maxlen = maxlength,padding = 'post',value = -1)
	#(sample,timestep,dimensions of data points)（样本数、时间步长、样本点本身的维度）
	data_X = np.reshape(data_X,(data_X.shape[0],maxlength,1))
	data_batch_X = data_X / float(2*num_questions)
	data_batch_y = np_utils.to_categorical(dataY,2*num_questions)
	return data_batch_X,data_batch_y

def rnn_keras_model_train(train_data,test_data,num_questions):
	interaction_length = []
	train_data_xset = []
	train_data_yset = []
	test_data_xset = []
	test_data_yset = []

	for userdata in train_data:
		train_data_xset.append(userdata.history)
		train_data_yset.append(userdata.next_answer[-1])
		interaction_length.append(userdata.length)
	for userdata in test_data:
		test_data_xset.append(userdata.history)
		test_data_yset.append(userdata.next_answer[-1])
		interaction_length.append(userdata.length)
	maxlength = max(interaction_length)
	LOGGER.info('the number of students--train/test:{}/{}'.format(len(train_data_xset),len(test_data_xset)))
	LOGGER.info('the total number of knowledge components:{}'.format(num_questions))
	LOGGER.info('the maxlength of student interactions:{}'.format(maxlength))

	'''
	def train_data_generator(train_dataset):
		for userdata in train_dataset:
			train_data_x = []
			train_data_y = []
			train_data_x.append(userdata.history)
			train_data_y.append(userdata.num_questions[-1])
			data_X = pad_sequences(train_data_x,maxlen = maxlength,padding = 'post',value = -1)
			#(sample,timestep,dimensions of data points)（样本数、时间步长、样本点本身的维度）
			data_X = np.reshape(data_X,(data_X.shape[0],maxlength,1))
			X = data_X / float(2*num_questions)
			y = np_utils.to_categorical(train_data_y,2*num_questions)
			yield (X,y)
	'''
	# data_X = pad_sequences(train_data_xset,maxlen = maxlength,padding = 'post',value = -1)
	# #(sample,timestep,dimensions of data points)（样本数、时间步长、样本点本身的维度）
	# data_X = np.reshape(data_X,(data_X.shape[0],maxlength,1))
	# X = data_X / float(2*num_questions)
	# y = np_utils.to_categorical(train_data_yset,2*num_questions)

	model = Sequential()
	model.add(Masking(mask_value = -1, input_shape=(maxlength, 1)))
	model.add(LSTM(2*num_questions,activation='softmax'))
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss= my_loss, optimizer=sgd, metrics=[my_accuracy])
	# model.fit_generator(train_data_generator(train_data),steps_per_epoch = len(train_data),epochs = 5,verbose = 2)
	
	test_X,test_y = data_standardization(test_data_xset,test_data_yset,maxlength,num_questions)
	k = 0
	num_samples = 50
	for index in range(0,len(train_data_xset),num_samples):
		k = k+1
		LOGGER.info("batch:{}".format (k))
		train_X,train_y = data_standardization(train_data_xset[index:index+num_samples],
								   train_data_yset[index:index+num_samples],
								   maxlength,num_questions)
		model.fit(train_X, train_y, epochs=2, batch_size=25, verbose=2, shuffle = True)

		train_scores = model.evaluate(train_X, train_y, verbose=0)
		LOGGER.info("Model Accuracy on the trianset: {}%".format(round(train_scores[1]*100,2)))
		test_scores = model.evaluate(test_X,test_y,verbose=0)
		LOGGER.info("Model Accuracy on the testset: {}%".format(round(test_scores[1]*100,2)))

	return model
	# plot_model(model,to_file = 'model.png',show_shapes = True)
	# model.save('keras_rnn_model.h5')

	'''
	x = [419,420,421,3,5]
	x = pad_sequences([x],maxlen = maxlength,padding = 'post',value = -1)
	x = np.reshape(x, (1,maxlength,1))
	x = x / float(2 * num_questions)
	prediction = model.predict(x, verbose=0)
	LOGGER.info([(prediction[0][i] / (prediction[0][i-num_questions]+prediction[0][i])) for i in range(num_questions,2*num_questions,1)])
	'''

def rnn_keras(data_folds,num_questions,seed,use_correct=True, use_hints=False,which_fold = None):
    if which_fold is not None and not (1 <= which_fold <= num_folds):
       raise ValueError("which_fold ({which_fold}) must be between 1 "\
                         "and num_folds({num_folds})".format(which_fold=which_fold,num_folds=num_folds))
    np.random.seed(seed)
    for fold_num, (train_data, test_data) in enumerate(data_folds):
        fold_num += 1
        if which_fold and fold_num != which_fold:
            continue
        LOGGER.info("Beginning fold %d", fold_num)
        train_nn_data = build_nn_data(train_data, num_questions, use_correct, use_hints)
        test_nn_data = build_nn_data(test_data, num_questions, use_correct, use_hints)
       	rnn_model = rnn_keras_model_train(train_nn_data,test_nn_data,num_questions)
       	model_file_path = 'model/'+'rnn_keras_model_'+str(fold_num)+'.h5'
       	rnn_model.save(model_file_path)

    LOGGER.info("Completed all %d folds", fold_num)