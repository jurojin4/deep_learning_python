import os
import pickle

dirname = os.path.dirname(__file__)

with open(os.path.join(dirname, "logs.pickle"), "rb") as file:
    dictionary = pickle.load(file)
print(dictionary)

dictionary["batch_size"] = 32
dictionary["warmup_epoch"] = 4
dictionary["data_augmentaion"] = False
dictionary["save_metric"] = "mAP@50"

with open(os.path.join(dirname, "logs.pickle"), "wb") as file:
    pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)