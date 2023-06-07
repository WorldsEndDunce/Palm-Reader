import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
import ast
from torch._dynamo.symbolic_convert import explain
from tf_explain.core.smoothgrad import SmoothGrad

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
frame_selection_rate = 3 # Select 1 frame every frame_selection_rate frames

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
                start_index = num_frames // 4  # Start from the 25th percentile
                end_index = num_frames - num_frames // 4  # End at the 75th percentile
                # Apply frame selection to the middle 50% frames
                cur = np.concatenate([cur[:start_index], cur[start_index:end_index:frame_selection_rate], cur[end_index:]], axis=0)
                examples.append(cur)
                index += cur.shape[0]
                for _ in range(cur.shape[0]):
                    action_indices.append(actions_dict[name])

data = np.concatenate(examples, axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

y_data = to_categorical(action_indices, num_classes=(len(actions)))`

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=420)

# Hyperparameter search
param_grid = {
    'num_heads': [4],
    'learning_rate': [0.001],
    "num_layers": [1, 3]
    # 'reg_strength': [0.001, 0.01, 0.1],
}

# Initialize a dictionary to store training history
history_dict = {}
# Initialize a list to store confusion matrices
confusion_matrices = []

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
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150,
                        callbacks=[ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                                   save_best_only=True, mode='auto'),
                                   ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                     mode='auto')])

    # Store the training history for this hyperparameter combination
    history_dict[str(params)] = history.history

    # Predict labels for the validation set
    y_pred = model.predict(x_val)
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Compute and store the confusion matrix
    confusion_matrix_val = confusion_matrix(y_true, y_pred)
    confusion_matrices.append(confusion_matrix_val)

    # Generate and save saliency maps (broken)
    # images_subset = x_val[:10]  # Adjust the number of images to generate saliency maps for
    # explainer = SmoothGrad()
    # explainer_args = {
    #     "num_samples": 50,
    #     "noise": 0.20
    # }
    #
    # for i, image in enumerate(images_subset):
    #     image = image[:, :, np.newaxis]  # Reshape image to (height, width, channels=1)
    #     class_index = np.argmax(model.predict(image[np.newaxis, ...]))  # Get predicted class index
    #     saliency = explainer.explain(validation_data=(image[np.newaxis, ...], None), model=model, class_index=class_index, **explainer_args)
    #
    #     # Plot and save the saliency maps
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     axes[0].imshow(image.squeeze(), cmap='gray')
    #     axes[0].set_title("Original Image")
    #     axes[0].axis("off")
    #
    #     axes[1].imshow(saliency.squeeze(), cmap='jet')
    #     axes[1].set_title("Saliency Map")
    #     axes[1].axis("off")
    #
    #     plt.savefig(f"saliency_map_{i}.png")
    #     plt.close()

# Print the training history for each hyperparameter combination
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
    params_dict = ast.literal_eval(params)
    plt.savefig(os.path.join("figures", "loss_acc_" + 'lr='+str(params_dict["learning_rate"])+'heads='+str(params_dict["num_heads"])+'layers='+str(params_dict["num_layers"]) + ".png"))
    plt.close()

# Plot the confusion matrix
for i, matrix in enumerate(confusion_matrices):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=actions, yticklabels=actions)
    params = list(ParameterGrid(param_grid)[i])
    params_dict = param_grid
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Hyperparameters: " + str(params_dict))
    plt.savefig(os.path.join("figures", "confusion_" + 'lr='+str(params_dict["learning_rate"])+'heads='+str(params_dict["num_heads"])+'layers='+str(params_dict["num_layers"]) + ".png"))
    plt.close()

# TODO: Add other plots?


