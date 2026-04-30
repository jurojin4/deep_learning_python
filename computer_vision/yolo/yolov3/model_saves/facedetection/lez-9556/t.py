import os
import torch
import pickle
dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "bboxes_prior.pickle"), "rb") as file:
    print(torch.tensor(pickle.load(file)))