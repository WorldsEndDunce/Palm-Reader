import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from models import init_model

# Enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions_dict = {"D0X": 0, "B0A": 1, "B0B": 2, "G01": 3, "G02": 4, "G03": 5, "G04": 6, "G05": 7, "G06": 8, "G07": 9, "G08": 10, "G09": 11, "G10": 12, "G11": 13}


# List of possible gestures
actions = [ # ids 1-14, respectively
    "Non-gesture",
    "Pointing with one finger",
    "Pointing with two fingers",
    "Click with one finger",
    "Click with two fingers",
    "Throw up",
    "Throw down",
    "Throw left",
    "Throw right",
    "Open twice",
    "Double click with one finger",
    "Double click with two fingers",
    "Zoom in",
    "Zoom out"
]
print(len(actions))

# Create data from pre-processed examples (not in this repo b/c too large)
folder_path = "dataset"
examples = []
action_indices = []
index = 0
for filename in os.listdir(folder_path):

    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".npy"):
        split_file = filename.split("_")
        name = split_file[1]
        #print(actions_dict[name])
        cur = np.load(file_path)
        if not cur.all():
            #print(cur.shape)
            if cur.ndim == 3:
                examples.append(cur)
                index += cur.shape[0]
                for _ in range(cur.shape[0]):
                    action_indices.append(actions_dict[name])

# print(action_indices)
data = np.concatenate(examples, axis=0)

# print(data.shape)
# print(index)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

# print(x_data.shape)
# print(labels.shape)
# print(np.max(labels))
# print(np.unique(labels))
# print(len(action_indices))

y_data = to_categorical(action_indices, num_classes=(len(actions)))
# print(y_data.shape)

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

# split data into training and testing
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=420)

# # initialize model
# model = init_model(x_train, actions, "Transformer")
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# model.summary()

# Define the hyperparameters to search
param_grid = {
    'num_heads': [3, 4, 6],
    'learning_rate': [0.001],
    # 'reg_strength': [0.001, 0.01, 0.1],
    # Add other hyperparameters to search
}

# Initialize a dictionary to store training history
history_dict = {}

# Iterate over each hyperparameter combination and fit the model
for params in ParameterGrid(param_grid):
    print("Training model with hyperparameters:", params)

    # Create a new instance of the model
    model = init_model(x_train, actions, "Transformer", params["num_heads"])

    # Compile and update learning rate
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

    model.optimizer.learning_rate.assign(params['learning_rate'])
    model_path = 'models/model_' + 'lr='+str(params["learning_rate"])+'heads='+str(params["num_heads"])+'.h5'
    # Fit the model to the data
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50,
                        callbacks=[ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                                   save_best_only=True, mode='auto'),
                                   ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                     mode='auto')])

    # Store the training history for this hyperparameter combination
    history_dict[str(params)] = history.history
    # save model

# Print the training hi story for each hyperparameter combination
for params, history in history_dict.items():
    print("Training history for hyperparameters:", params)
    print(history)
    # plot training loss and accuracy
    fig, loss_ax = plt.subplots(figsize=(16, 10))
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history['loss'], 'y', label='train loss')
    loss_ax.plot(history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history['acc'], 'b', label='train acc')
    acc_ax.plot(history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')

    plt.show()

# TODO: Add other plots? Looking at you, Dishwison


