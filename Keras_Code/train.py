from data.dataloader import DataLoader
from options import Options
from model.nets import Net

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import pairwise_distances
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model

import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def get_optimizer(optimizer_name, lr):
    if(optimizer_name == 'rmsprop'):
        return RMSprop(lr=lr)
    elif(optimizer_name == 'adam'):
        return Adam(lr=lr)

def extract_genuines_impostors(distances, labels, sessions, sort=True):
    # Get genuine matching and impostor matching scores
    num_features = distances.shape[0]
    genuines = []
    impostors = []

    for i in range(num_features-1):
        for j in range(i+1, num_features):
                
            if(sessions[i]!=sessions[j]):
                if(labels[i]==labels[j]):
                    genuines.append(distances[i, j])
                else:
                    impostors.append(distances[i, j])
    
    if sort:
        genuines = sorted(genuines)
        impostors = sorted(impostors)
        
    return np.array(genuines), np.array(impostors)

def plot_legend(loc):
    legend = plt.legend(loc=loc, shadow=False, prop={'size': 10})
    legend.get_frame().set_facecolor('#ffffff')  
    
def plot_DET(frr, far, linthresh, output_path):
    plt.figure()
    scale_type='symlog'
    plt.xscale(scale_type, linthresh=linthresh)
    plt.yscale(scale_type, linthresh=linthresh)
    plt.plot(frr, far, linestyle='-', linewidth=1, label=config.model_name)
    plt.xlabel('False Rejected Rate(%)')
    plt.ylabel('False Accepted Rate(%)')
    plot_legend('best')
    plt.savefig(output_path + '_DET.png')

def plot_MDD(genuines, impostors, output_path):
    # Produce matching distance distributions
    df1 = pd.DataFrame(genuines, columns = ['GenuineScores'])
    df1.GenuineScores.plot.kde(label='Genuine')
    df2 = pd.DataFrame(impostors, columns = ['ImpostorScores'])
    df2.ImpostorScores.plot.kde(label='Impostor')
    
    # Plot matching distance distributions
    plt.figure()
    plt.xlabel('Distance')
    plt.ylabel('Probability Density')
    plot_legend('best')
    plt.savefig(output_path + 'MDD.png')
    
def convert_to_TFLite(keras_model, file_path=''):
    
    print("Converting to TFLite...", end='\r')
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(file_path,'wb').write(tflite_model)
    print("Converted to TFLite.")
    
    return tflite_model

def test(model, fold, n_sessions=1):
    
    fname = dataloader.test_name + '_fold' + str(fold)
    fpath = config.output_folder + fname
    scores_path  = fpath + '_scores.txt'
    feature_path = fpath + ".hdf5"
    
    print("Evaluate fold %i on %s database:"%(fold, dataloader.test_name))
    print("Get features.")
    
    if(model!=None):

        inputs  = model.inputs[0]
        outputs = model.outputs
        get_output = K.function(inputs,outputs)
        n = outputs[0].shape[-1]
        features = np.zeros((dataloader.n_test_samples, n), dtype="float32")
        labels = dataloader.test_labels
        sessions = dataloader.test_sessions

        for i in range(0, dataloader.n_test_samples, dataloader.test_class_size):
            j = i+dataloader.test_class_size
            batch = dataloader.test_data[i:j]
            fs = np.array(get_output(batch))[0]
            features[i:j] = fs
       
        # save features and labels
        with h5py.File(feature_path, 'w') as h5f_data:
            h5f_data.create_dataset("features", data=features, maxshape=(None, features.shape[1]), chunks=True)
            h5f_data.create_dataset("labels", data=dataloader.test_labels, maxshape=(None,), chunks=True) 
            h5f_data.create_dataset("sessions", data=dataloader.test_sessions, maxshape=(None,), chunks=True) 
    else:
        with h5py.File(feature_path, 'r') as h5f_data:
            features = np.array(h5f_data["features"])
            labels = np.array(h5f_data["labels"])
            sessions = np.array(h5f_data["sessions"])
            
    # VERIFICATION
    
    print ("Verification using pairwise cosine distance.")
    distances = pairwise_distances(features, Y=None, metric=config.distance_metric)
    genuines, impostors = extract_genuines_impostors(distances, labels, sessions, True)
    
    far = []
    frr = [] 

    # generate thresholds
    n_thresholds = 1000
    epsilon = 1e-5
    start = min(0, np.amin(genuines)-epsilon)
    end = np.amax(impostors)+epsilon
    threshold_step = (end-start)/n_thresholds
    thresholds=np.arange(start, end, threshold_step)
    
    # calculate FAR and FRR
    n_genuines = len(genuines)
    n_impostors = len(impostors)

    far = np.searchsorted(impostors, thresholds, side='right') * 100 / n_impostors
    frr = np.searchsorted(genuines, thresholds, side='right')*(-100) / n_genuines + 100.0

    # calculate the most optimal FAR and FRR values => EER
    k = np.argmin(np.abs(far-frr))
    eer = (far[k]+frr[k])/2

    # write verification rate to file
    with open(scores_path, 'a+') as file:
        file.write("\nThreshold: {:.3f}, FRR: {:.3f}, FAR: {:.3f}, EER: {:.3f}\n\n".format(thresholds[k],frr[k],far[k],eer))

    print('Verification results:')
    print("Threshold: %.3f, FRR: %.3f, FAR: %.3f, EER: %.3f"%(thresholds[k], frr[k], far[k], eer))
    
    # Plot DET
    plot_DET(frr, far, linthresh=20, output_path=fpath)
    
    # Plot matching distance distributions
    plot_MDD(genuines, impostors, output_path=fpath)
    
    return eer

def train(
    retrain_softmax=True, 
    retrain_fine_tune=True,
    convert_to_tflite=True,
    n_sessions=1
):
    best_eer = 100.0
    best_eer_fold = 1
    avg_eer = 0.0
    
    net = Net(config)
    
    for fold in range(config.n_folds):
        train_data, train_labels, valid_data, valid_labels = dataloader.get_fold_data(fold)
        fold+=1
        
        if(retrain_softmax==False and retrain_fine_tune==False):
            net.adacos_model = None
        else:
         
            print('Fitting fold #%i...'%fold)
            K.clear_session()
            train_labels_float = train_labels.astype(float)
            valid_labels_float = valid_labels.astype(float)
            add_embedding=True
            
            # Get model
            if(config.model_name=='lawnet'):
                net.build_lawnet_backbone(input_shape = dataloader.sample_shape)
                add_embedding=False
            elif(config.model_name=='mpsnet'):
                net.build_mpsnet_backbone(input_shape = dataloader.sample_shape)
                add_embedding=False
            elif(config.model_name=='cnn1'):
                net.build_cnn1_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='cnn2'):
                net.build_cnn2_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='vgg16'):
                net.build_vgg16_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='mobilenet_v1'):
                net.build_mobilenet_v1_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='mobilenet_v2'):
                net.build_mobilenet_v2_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='mobilenet_v3'):
                net.build_mobilenet_v3_backbone(input_shape = dataloader.sample_shape)
            elif(config.model_name=='mobilefacenet'):
                net.build_mobilefacenet_backbone(input_shape = dataloader.sample_shape)
                
            net.build_softmax_model(n_classes=dataloader.n_train_classes, add_embedding=add_embedding)
            optimizer = get_optimizer(config.warmup_optimizer, config.warmup_lr)
            net.softmax_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
            postfix = config.output_folder + 'softmax_fold' + str(fold)
            best_weight = postfix + '.hdf5'

            if(retrain_softmax):
                net.softmax_model.summary()
                
                print("First phase: Fitting model with Softmax loss...\n")
                
                checkpoint = ModelCheckpoint(best_weight, verbose=1, save_best_only=True)
                
                history = net.softmax_model.fit(train_data, train_labels_float, 
                                                validation_data=(valid_data, valid_labels_float), 
                                                epochs=config.warmup_epochs, 
                                                batch_size=config.warmup_batch_size,
                                                class_weight=dataloader.class_weights,
                                                callbacks=[checkpoint])
                
                history_path = postfix + '_history.pkl'

                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)

            if(retrain_fine_tune):
                net.softmax_model.load_weights(best_weight)
                softmax_valid_scores = net.softmax_model.evaluate(valid_data, valid_labels_float, verbose=0)
                print('Fold #%i validation scores (Softmax): '%fold, softmax_valid_scores)

            ### Fine tune with AdaCos
            postfix = config.output_folder + 'adacos_fold' + str(fold)
            best_weight =  postfix + '.hdf5'
            
            # Get fine tune model
            net.build_adacos_model()
            optimizer = get_optimizer(config.fine_tune_optimizer, config.fine_tune_lr)
            net.adacos_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        
            if(retrain_fine_tune):
                net.adacos_model.summary()
                
                print("Second phase: Fine tune the best softmax model with AdaCos...")
                
                checkpoint = ModelCheckpoint(best_weight, monitor='val_loss', verbose=1, save_best_only=True)
                history = net.adacos_model.fit([train_data, train_labels], train_labels_float, 
                                                batch_size=config.fine_tune_batch_size, 
                                                epochs=config.fine_tune_epochs, 
                                                validation_data=([valid_data, valid_labels], valid_labels_float), 
                                                class_weight=dataloader.class_weights,
                                                callbacks=[checkpoint])
                history_path = postfix + '_history.pkl'

                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)

            net.adacos_model.load_weights(best_weight)
            adacos_valid_scores = net.adacos_model.evaluate([valid_data, valid_labels], valid_labels_float, verbose=0)
            print('Fold #%i validation scores (AdaCos): '%fold, adacos_valid_scores)
        
        # Convert model to TensorFlow-Lite version
        if(net.adacos_model!=None):
            tflite_file_path = postfix + '.tflite'
            net.adacos_model = Model(inputs=net.adacos_model.inputs[0], outputs=net.adacos_model.get_layer(config.embedding_layer_name).output)
        
            if(convert_to_tflite):
                convert_to_TFLite(net.adacos_model, tflite_file_path)
                
        # Get identification rate (accuracy) and verification rate EER
        eer = test(net.adacos_model, fold, n_sessions=n_sessions)
        
        # update eer
        avg_eer += eer
            
        if(eer<best_eer):
            best_eer=eer
            best_eer_fold=fold

    # calculate average eer
    avg_eer /= config.n_folds

    # write test scores to file
    scores_path = config.output_folder + dataloader.test_name + '_scores.txt'
    
    with open(scores_path, "w") as f:
        f.write("Best EER fold: {:d}\nBest EER: {:.6f}\nAverage EER: {:.6f}".format(best_eer_fold, best_eer, avg_eer))
    
    print("Best EER fold: %i\nBest EER: %.6f\nAverage EER: %.6f"%(best_fold, best_eer, avg_eer))

if __name__ == '__main__':
    
    config = Options().parse()
    dataloader = DataLoader(config)
    retrain_softmax   = True
    retrain_fine_tune = True
    convert_to_tflite = True

    for line in config.test_folders:
        parts = line.strip().split(' ')
        test_folder, n_sessions = parts[0], int(parts[1])
        dataloader.load_test_data(test_folder = test_folder)
        train(retrain_softmax, retrain_fine_tune, convert_to_tflite, n_sessions=n_sessions)
        retrain_softmax   = False
        retrain_finetune  = False
        convert_to_tflite = False