########################################################
#Project name: Multiverse
#Date: 2024
########################################################
from __future__ import division


import os
import csv
import argparse
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import random
from time import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation,multiply,MaxPooling1D,Add,AveragePooling1D,Lambda,Permute
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from keras.regularizers import l2

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize

from ModelHandler import add_model,load_model_structure, ModelHandler
from custom_metrics import *
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as precision_recall_fscore
from tensorflow.keras import backend as K

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
# tf.random_set_seed(seed)
np.random.seed(seed)
random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def save_npz(path,train_name,train_data,val_name,val_data):
    check_and_create(path)
    np.savez_compressed(path+train_name, train=train_data)
    np.savez_compressed(path+val_name, val=val_data)

def custom_label(y, strategy='one_hot'):
    'This function generates the labels based on input strategies, one hot, reg'
    print('labeling stragey is', strategy,y)
    y_shape = y.shape
    num_classes = y_shape[1]
    if strategy == 'one_hot':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # if any(t < 0 for t in thisOutputs):
            #     print("thisOutputs",thisOutputs)
            #     print(np.where(thisOutputs <0 ))
            #     a = np.where(thisOutputs == np.amax(thisOutputs[np.where(thisOutputs <0 )]))
            #     print("a,v",a)

                # print(thisOutputs.argsort())
                # for a in thisOutputs.argsort():
                #     if thisOutputs[a]==0:
                #         print(a-1)


            # logOut = 20*np.log10(thisOutputs)
            # print("type",type(thisOutputs),thisOutputs)
            if np.any(thisOutputs<0):
                max_index = np.where(thisOutputs == np.amax(thisOutputs[np.where(thisOutputs <0 )]))
            else:
                max_index = thisOutputs.argsort()[-1:][::-1]  # For one hot encoding we need the best one

            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # print("thisOutputs",y[i,:])
            # # if any(t < 0 for t in thisOutputs):
            # #     print(thisOutputs)
            # thisOutputs_arrange = [i[0] for i in sorted(enumerate(thisOutputs), key=lambda k: k[1], reverse=True)]
            # # print(thisOutputs_arrange,thisOutputs[thisOutputs_arrange[0]])

            # if any(t < 0 for t in thisOutputs):
            #     print("thisOutputs",thisOutputs)
            #     negative_indexes = np.where(thisOutputs <0 )
            #     print("negative_indexes",negative_indexes)
            #     print("vals",thisOutputs[negative_indexes])

            # logOut = 20*np.log10(thisOutputs)   # old version
            # for a in range(len(thisOutputs)):
            #     if a<0:
            #         thisOutputs[a]+=300


            logOut = (thisOutputs-min(thisOutputs))/(-1*min(thisOutputs)+1)
            # logOut = thisOutputs#-min(thisOutputs)
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y,num_classes


def over_k(true,pred):
    dicti = {}
    for kth in range(100):
        kth_accuracy = metrics.top_k_categorical_accuracy(true,pred,k=kth)
        with tf.Session() as sess: this = kth_accuracy.eval()
        dicti[kth] =this
    return dicti


def precison_recall_F1(model,Xtest,Ytest):
    #####For recall and precison
    y_pred = model.predict(Xtest)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_true_bool = np.argmax(Ytest, axis=1)
    return precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted')


def detecting_related_file_paths(path,categories,episodes):
    find_all_paths =['/'.join(a.split('/')[:-1]) for a in show_all_files_in_directory(path,'rf.npz')]     # rf for example
    selected = []
    for Cat in categories:   # specify categories as input
        for ep in episodes:
            selected = selected + [s for s in find_all_paths if Cat in s.split('/') and 'episode_'+str(ep) in s.split('/')]
    print('Getting {} data out of {}'.format(len(selected),len(find_all_paths)))

    return selected


def get_data(data_paths,modality,key,train_catergory,aug_category,percentage_WI):   # per cat for now, need to add per epside for FL part
    #####gets train catergory and return test sets for all catergories
    for l in tqdm(data_paths):
        randperm = np.load(l+'/ranperm.npy')
        open_file = open_npz(l+'/'+modality+'.npz',key)
        print("test",l,'/'.join(l.split('/')[:4])+'/flash_WI_labels/'+'/'.join(l.split('/')[5:-1])+'/npz/'+modality+'.npz')
        if modality=='rf':
            open_file_WI = open_npz('/'.join(l.split('/')[:4])+'/flash_WI_labels/'+'/'.join(l.split('/')[5:-1])+'/npz/'+modality+'.npz',key)
        else:
            open_file_WI = open_file

        if train_catergory[0] in l.split('/'):
            try:
                train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
                validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)
            except NameError:
                train_data = open_file[randperm[:int(0.8*len(randperm))]]
                validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]

        if aug_category[0] in l.split('/'):
            try:
                train_data = np.concatenate((train_data, open_file_WI[randperm[:int(percentage_WI*0.8*len(randperm))]]),axis = 0)
                validation_data = np.concatenate((validation_data, open_file_WI[randperm[int(0.8*len(randperm)):int(0.8*len(randperm)+percentage_WI*0.1*len(randperm))]]),axis = 0)
            except NameError:
                train_data = open_file_WI[randperm[:int(percentage_WI*0.8*len(randperm))]]
                validation_data = open_file_WI[randperm[int(0.8*len(randperm)):int(0.8*len(randperm)+percentage_WI*0.1*len(randperm))]]


        ####PER CAT
        if 'Cat1' in l.split('/'):
            try:
                test_data_cat1 = np.concatenate((test_data_cat1, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat1 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat2' in l.split('/'):
            try:
                test_data_cat2 = np.concatenate((test_data_cat2, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat2 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat3' in l.split('/'):
            try:
                test_data_cat3 = np.concatenate((test_data_cat3, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat3 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat4' in l.split('/'):
            try:
                test_data_cat4 = np.concatenate((test_data_cat4, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat4 = open_file[randperm[int(0.9*len(randperm)):]]
    print("shapes",train_data.shape, validation_data.shape, test_data_cat1.shape, test_data_cat3.shape)
    return train_data, validation_data, test_data_cat1, test_data_cat1, test_data_cat3, test_data_cat3




parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')

parser.add_argument('--epochs', default=100, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)

parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/twin_flash/baseline_code/models')
parser.add_argument('--image_feature_to_use', type=str ,default='raw', help='feature images to use',choices=['raw','custom'])

parser.add_argument('--experiment_catergories', nargs='*' ,default=['Cat1','Cat3'], help='categories included',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--experiment_epiosdes', nargs='*' ,default=['0','1','2','3','4','5','6','7','8','9'], help='episodes included',choices=['0','1','2','3','4','5','6','7','8','9'])
parser.add_argument('--training_catergory',nargs='*', default=['Cat1'], help='categories for training',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--aug_catergory',nargs='*', default=['Cat3'], help='categories for training',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--aug_ratio', default=0.1, type=float,help='augmentaion ratio',)

args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

check_and_create(args.model_folder)


print('******************Detecting related file paths*************************')
selected_paths = detecting_related_file_paths(args.data_folder,args.experiment_catergories,args.experiment_epiosdes)
###############################################################################
# Outputs ##needs to be changed
###############################################################################
print('******************Getting RF data*************************')
# RF_train, RF_val,RF_c1,RF_c2,RF_c3,RF_c4 = get_data(selected_paths,'rf','rf',args.training_catergory)
RF_train, RF_val,RF_c1,RF_c2,RF_c3,RF_c4 = get_data(selected_paths,'rf','rf',args.training_catergory,args.aug_catergory,args.aug_ratio)
print('RF data shapes on same client',RF_train.shape,RF_val.shape)

y_train,num_classes = custom_label(RF_train,args.strategy)
y_validation, _ = custom_label(RF_val,args.strategy)
# print("y_validation",y_validation)
y_c1, _ = custom_label(RF_c1,args.strategy)
y_c2, _ = custom_label(RF_c2,args.strategy)
y_c3, _ = custom_label(RF_c3,args.strategy)
y_c4, _ = custom_label(RF_c4,args.strategy)
print("y_train",y_train,y_train.shape)
# print(marmar)
###############################################################################
# Inputs ##needs to be changed
###############################################################################

if 'coord' in args.input:
    print('******************Getting Gps data*************************')
    X_coord_train, X_coord_validation,gps_c1,gps_c2,gps_c3,gps_c4 = get_data(selected_paths,'gps','gps',args.training_catergory,args.aug_catergory,args.aug_ratio)
    print('GPS data shapes',X_coord_train.shape,X_coord_validation.shape)
    coord_train_input_shape = X_coord_train.shape

    ### normalize
    X_coord_train = X_coord_train / 9747
    X_coord_validation = X_coord_validation / 9747
    gps_c1 = gps_c1/ 9747
    gps_c2 = gps_c2/ 9747
    gps_c3 = gps_c3/ 9747
    gps_c4 = gps_c4/ 9747
    ## For convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    gps_c1 = gps_c1.reshape((gps_c1.shape[0], gps_c1.shape[1], 1))
    gps_c2 = gps_c2.reshape((gps_c2.shape[0], gps_c2.shape[1], 1))
    gps_c3 = gps_c3.reshape((gps_c3.shape[0], gps_c3.shape[1], 1))
    gps_c4 = gps_c4.reshape((gps_c4.shape[0], gps_c4.shape[1], 1))

# print(marmar)
if 'img' in args.input:
    print('******************Getting image data*************************')
    X_img_train, X_img_validation,img_c1,img_c2,img_c3,img_c4 = get_data(selected_paths,'image','img',args.training_catergory,args.aug_catergory,args.aug_ratio)
    print('image data shapes',X_img_train.shape,X_img_validation.shape)
    ###normalize images
    X_img_train = X_img_train / 255
    X_img_validation = X_img_validation / 255
    img_c1 = img_c1/ 255
    img_c2 = img_c2/ 255
    img_c3 = img_c3/ 255
    img_c4 = img_c4/ 255
    img_train_input_shape = X_img_train.shape

if 'lidar' in args.input:
    print('******************Getting lidar data*************************')
    X_lidar_train, X_lidar_validation,lid_c1,lid_c2,lid_c3,lid_c4 = get_data(selected_paths,'lidar','lidar',args.training_catergory,args.aug_catergory,args.aug_ratio)
    print('lidar data shapes',X_lidar_train.shape,X_lidar_validation.shape)
    lidar_train_input_shape = X_lidar_train.shape

print('******************Succesfully generated the data*************************')
##############################################################################
# Model configuration
##############################################################################
print('******************Configuring the models*************************')
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True

modelHand = ModelHandler()
opt = Adam(lr=args.lr,amsgrad=True)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# opt = SGD(lr=args.lr)
if 'coord' in args.input:
    if args.restore_models:
        coord_model = load_model_structure(args.model_folder+'coord_model.json')
        coord_model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)
        # coord_model.trainable = False
    else:
        coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy, fusion)
        if not os.path.exists(args.model_folder+'coord_model.json'):
            add_model('coord',coord_model,args.model_folder)
if 'img' in args.input:
    if args.image_feature_to_use == 'raw':
        model_type = 'raw_image'
    elif args.image_feature_to_use == 'custom':
        model_type = 'custom_image'

    if args.restore_models:
        img_model = load_model_structure(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json')
        img_model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)
        # img_model.trainable = False
    else:
        img_model = modelHand.createArchitecture(model_type,num_classes,[img_train_input_shape[1],img_train_input_shape[2],3],'complete',args.strategy,fusion)
        if not os.path.exists(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json'):
            add_model('image_'+args.image_feature_to_use,img_model,args.model_folder)

if 'lidar' in args.input:
    if args.restore_models:
        lidar_model = load_model_structure(args.model_folder+'lidar_model.json')
        lidar_model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)
        # lidar_model.trainable = False
    else:
        lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)
        if not os.path.exists(args.model_folder+'lidar_model.json'):
            add_model('lidar',lidar_model,args.model_folder)

##############################################################################
# Fusion: Coordinate+Image+LIDAR
###############################################################################
if multimodal == 3:
    print("***********Fusion of all modalities****************")
    x_train = [X_lidar_train,X_img_train,X_coord_train]
    x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
    combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output])
    reg_val=0.001
    z = Dense(1024,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
    z = BatchNormalization()(z)
    z = Dense(512,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    z = Dense(256,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    z = Dense(128,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    z = Dense(num_classes, activation="relu",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    add_model('coord_img_raw_lidar',model,args.model_folder)   #add fusion model
    # model.compile(loss=categorical_crossentropy,
    #               optimizer=opt,
    #               metrics=[metrics.categorical_accuracy,
    #                                     top_2_accuracy, top_5_accuracy,top_10_accuracy,top_25_accuracy,top_50_accuracy])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=opt,
                  metrics=[metrics.mae, metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy,top_25_accuracy,top_50_accuracy])
    model.summary()
    print("x_train,y_train",len(x_train),len(y_train))
    hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_lidar1_'+args.image_feature_to_use+'_trained_on'+args.training_catergory[0]+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

    # print(hist.history.keys())
    # print('loss',hist.history['loss'],'val_loss',hist.history['val_loss'],'categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
    #             ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

    print('***************Testing the model************')
    model.load_weights(args.model_folder+'best_weights.coord_img_lidar1_'+args.image_feature_to_use+'_trained_on'+args.training_catergory[0]+'.h5', by_name=True)

    print('***************Testing per category************')
    x_c1 = [lid_c1, img_c1, gps_c1]
    x_c2 = [lid_c2, img_c2, gps_c2]
    x_c3 = [lid_c3, img_c3, gps_c3]
    x_c4 = [lid_c4, img_c4, gps_c4]
    scores_cat1 = model.evaluate(x_c1, y_c1)
    print('scores_cat1',scores_cat1)
    # scores_cat2 = model.evaluate(x_c2, y_c2)
    # print('scores_cat2',scores_cat2)
    scores_cat3 = model.evaluate(x_c3, y_c3)
    print('scores_cat3',scores_cat3)
    # scores_cat4 = model.evaluate(x_c4, y_c4)
    # print('scores_cat4',scores_cat4)
