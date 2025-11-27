import os
import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# The data is located in a different folder from the cwd
def get_path(input_filename, input_parent = "data"):
    """Get the file path for specified data file.

    Args:
        input_filename (str): Data file name in the ~/input_parent folder.
        input_parent (str): Parent folder name, defaults to 'data'.

    Returns:
        str: Full absolute path to file.
    """
    path_data = os.path.abspath(os.path.join('..', input_parent, input_filename))
    return path_data


def model_path(input_name):
    mod_loc = "models/"+input_name+".joblib"
    return get_path(mod_loc, 'outputs')


# def import_or_install(package):
#     # From StackOverflow,
#     # https://stackoverflow.com/a/62128239/23486987
#     try:
#         import package
#     except:
#         !pip install package
#         import package

def apply_model(input_set, input_model, input_window, input_threshold, input_return="adj"):

    pred_y_proba = input_model.predict_proba(input_set)[:,1]

    if input_return == "proba":
        return pred_y_proba

    pred_y_wt = medfilt(pred_y_proba, kernel_size = input_window)
    pred_y_wt = (pred_y_wt >= input_threshold).astype(np.int32)

    if input_return=="adj":
        return pred_y_wt

    if input_return=="all":

        # No window, default threshold
        pred_y_none = (pred_y_proba >= 0.5).astype(np.int32)

        # Best window, default threshold
        pred_y_w = medfilt(pred_y_proba, kernel_size=input_window)
        pred_y_w = (pred_y_w >= 0.5).astype(np.int32)

        # No window, best threshold
        pred_y_t = (pred_y_proba >= input_threshold).astype(np.int32)

        # # Best window, best threshold
        # pred_y_wt = medfilt(pred_y_proba, kernel_size = input_window)
        # pred_y_wt = (pred_y_wt >= input_threshold).astype(np.int32)
        return pred_y_none, pred_y_w, pred_y_t, pred_y_wt


def report_scores(input_true, input_pred, input_round=None):
    output_f1 = f1_score(input_true, input_pred)
    output_acc = accuracy_score(input_true, input_pred)
    output_pre = precision_score(input_true, input_pred)
    output_rec = recall_score(input_true, input_pred)

    if input_round is not None:
        output_f1 = round(output_f1, input_round)
        output_acc = round(output_acc, input_round)
        output_pre = round(output_pre, input_round)
        output_rec = round(output_rec, input_round)

    return output_f1, output_acc, output_pre, output_rec