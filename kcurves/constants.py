"""
Global constants for the device and data set.
"""
# TODO: Import the constants from scripts folder
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DATA_SET = "synthetic_lines"
#DATA_SET = "synthetic_functions"
DATA_SET = "synthetic_clusters"





