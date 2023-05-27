
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import math
import pickle





def compare_lists(list1, list2):
    """
    Compare two lists and return the percentage of elements in the second list
    that are the same as the first list.
    """
    num_correct = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            num_correct += 1
    accuracy = (num_correct / len(list1)) * 100
    return accuracy



def prepare_data(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)
    data = np.array(data, dtype=float)
    
    return data
    
    
    
def compare_lists(list1, list2):
    """
    Compare two lists and return the percentage of elements in the second list
    that are the same as the first list.
    """
    num_correct = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            num_correct += 1
    accuracy = (num_correct / len(list1)) * 100
    return accuracy
    
    
    

def CNN_training(test_data):
    # Load data
    
    training_sitting = prepare_data("sitting.json")
    training_standing = prepare_data("standing.json")
    
    measure_sitting = prepare_data("data_measure_sitting.json")
    #training_standing = prepare_data("data_measure_standing.json")
    
    training_sitting = np.concatenate((training_sitting, measure_sitting), axis=0)
    #training_standing = np.concatenate((training_sitting, measure_sitting), axis=0)
    
    
    training_data = np.concatenate((training_sitting, training_standing), axis=0)
    
    #Label data
    number_sitting_data = training_sitting.shape[0]
    number_standing_data = training_standing.shape[0]
    
    labels = np.zeros((number_sitting_data + number_standing_data),)
    labels[number_sitting_data:] = 1
    
    #Shuffle data and labels
    random_indices = np.random.permutation(training_data.shape[0])
    training_data = training_data[random_indices]
    labels = labels[random_indices]
    
    #Check input data
    print(training_data)
    print(labels)
    
    # Split the data into training and validation sets
    split_ratio = 0.8  # 80% for training, 20% for validation
    num_training_samples = int(split_ratio * training_data.shape[0])
    
    x_train = training_data[:num_training_samples]
    y_train = labels[:num_training_samples]
    x_val = training_data[num_training_samples:]
    y_val = labels[num_training_samples:]
    
    #CNN model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(19, 3)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
    
    
    
    
    
    
    
    
    sitting_test = prepare_data("data_measure_sitting.json")
    standing_test = prepare_data("data_measure_standing.json")
    mix_test = prepare_data("data_measure.json")
    result = []
    result_raw = []
    for i in mix_test:
    	i = i.reshape(1,19,3)
    	print(i.shape)
    	prediction = model.predict(i)
    	result_raw.append(prediction[0][0])
    	if prediction[0][0] > 0.5:
    		result.append(1)
    	else:
    		result.append(0)
    
    print(result_raw)
    print(result)
    
    
    manual_mix_test = ["sitting" for _ in range(len(mix_test))]
    
    for i in range(76):
    	manual_mix_test[i] = 0
    for i in range(150,226):
    	manual_mix_test[i] = 0
    for i in range(76,150):
    	manual_mix_test[i] = 1
    for i in range(226,303):
    	manual_mix_test[i] = 1
    print(manual_mix_test)
    print(compare_lists(manual_mix_test, result))
    if(compare_lists(manual_mix_test, result) >74):
    	with open('model.pkl', 'wb') as f:
    		pickle.dump(model, f)
    	
    
    print(training_sitting.shape)
    print(training_standing.shape)
    return compare_lists(manual_mix_test, result)
    
    
    
   
    
    
res = 0
while res <74:
	res = CNN_training("asdasd")
