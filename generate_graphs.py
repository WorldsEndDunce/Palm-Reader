import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import ast
'''
The graph function plots a loss/acc line graph for the training and validation sets,
a confusion matrix for predicted and true classifications, and a multi-class ROC curve.
These graphs are saved to the figures folder and titled with their respective hyperparameters.
'''
def graph(actions, history_dict, confusion_matrices, y_pred_all, y_true_all):
    # Print the training history for each hyperparameter combination
    for i, (params, history) in enumerate(history_dict.items()):
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
        file_name = 'loss_acc_' + 'lr=' + str(params_dict["learning_rate"]) + 'heads=' + str(
            params_dict["num_heads"]) + 'layers=' + str(params_dict["num_layers"]) + ".png"
        plt.savefig(os.path.join("figures", file_name))
        plt.close()

        # Plot the confusion matrix
        confusion_matrix = confusion_matrices[i]
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", xticklabels=actions, yticklabels=actions)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix for Hyperparameters: " + str(params_dict))
        file_name = 'confusion_' + 'lr=' + str(params_dict["learning_rate"]) + 'heads=' + str(
            params_dict["num_heads"]) + 'layers=' + str(params_dict["num_layers"]) + ".png"
        plt.savefig(os.path.join("figures", file_name))
        plt.close()

        # Iterate over each class and plot the ROC curve
        plt.figure(figsize=(12, 9))
        for j in range(len(actions)):
            fpr, tpr, _ = roc_curve((y_true_all[i] == j).astype(int), y_pred_all[i][:, j])
            roc_auc = auc(fpr, tpr)
            label = actions[j] + f" (AUC = {roc_auc:.2f})"
            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve')
        roc_file_name = 'roc_' + 'lr=' + str(params_dict["learning_rate"]) + 'heads=' + str(
            params_dict["num_heads"]) + 'layers=' + str(params_dict["num_layers"]) + ".png"
        plt.legend()
        plt.savefig(os.path.join("figures", roc_file_name))
        plt.close()