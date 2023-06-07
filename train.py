import numpy as np
import os

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from generate_graphs import graph
from models import init_model

# Enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions_dict = {"D0X": 0, "B0A": 1, "B0B": 2, "G01": 3, "G02": 4, "G03": 5, "G04": 6, "G05": 7, "G06": 8, "G07": 9, "G08": 10, "G09": 11, "G10": 12, "G11": 13}

# List of possible gestures
actions = [ # ids 1-14, respectively
    "Non-gesture",
    "Point w/ one finger",
    "Point w/ two fingers",
    "Click w/ one finger",
    "Click w/ two fingers",
    "Throw up",
    "Throw down",
    "Throw left",
    "Throw right",
    "Open twice",
    "2x click w/ one finger",
    "2x click w/ two fingers",
    "Zoom in",
    "Zoom out"
]

# Create data from pre-processed examples (not in this repo b/c too large)
print("Loading in data, please wait...")
folder_path = "dataset"
examples = []
action_indices = []
index = 0
frame_selection_rate = 10 # Select 1 frame every frame_selection_rate frames

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".npy"):
        split_file = filename.split("_")
        name = split_file[1]
        cur = np.load(file_path)
        if not cur.all():
            if cur.ndim == 3:
                # Calculate the indices for the middle 50% frames
                num_frames = cur.shape[0]
                start_index = num_frames // 4  # Start from the quarter mark
                end_index = num_frames - num_frames // 4  # End at the three quarters mark
                # Apply frame selection to the middle 50% frames
                cur = np.concatenate([cur[:start_index], cur[start_index:end_index:frame_selection_rate], cur[end_index:]], axis=0)
                examples.append(cur)
                index += cur.shape[0]
                for _ in range(cur.shape[0]):
                    action_indices.append(actions_dict[name])

data = np.concatenate(examples, axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

y_data = to_categorical(action_indices, num_classes=(len(actions)))

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=420)

# Hyperparameter search
param_grid = {
    'num_heads': [4],
    'learning_rate': [0.001],
    "num_layers": [1]
}

# Initialize a dictionary to store training history
history_dict = {}
# Initialize a list to store confusion matrices
confusion_matrices = []
# For ROC calculation
y_true_all = []
y_pred_all = []

# Iterate over each hyperparameter combination and fit the model
for params in ParameterGrid(param_grid):
    print("Training model with hyperparameters:", params)

    # Create a new instance of the model
    model = init_model(x_train, actions, "Transformer", params["num_heads"], params["num_layers"])

    # Compile and update learning rate
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

    model.optimizer.learning_rate.assign(params['learning_rate'])
    model_path = 'models/model_' + 'lr='+str(params["learning_rate"])+'heads='+str(params["num_heads"])+'layers='+str(params["num_layers"]) + '.h5'
    # Fit the model to the data
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1,
                        callbacks=[ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                                   save_best_only=True, mode='auto'),
                                   ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                     mode='auto')])

    # Store the training history for this hyperparameter combination
    history_dict[str(params)] = history.history

    # Predict labels for the validation set
    y_pred =np.argmax(model.predict(x_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Compute and store the confusion matrix
    confusion_matrix_val = confusion_matrix(y_true, y_pred)
    confusion_matrices.append(confusion_matrix_val)

    y_true_all.append(y_true)
    y_pred_all.append(model.predict(x_val))

# Plot and save performance graphs
graph(actions, history_dict, confusion_matrices, y_pred_all, y_true_all)




