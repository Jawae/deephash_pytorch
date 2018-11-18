from fastai.vision import *
from fastai import *

def run_folder(path, Learner):
    for i in (path).iterdir():
        Learner.predict(open_image(i))

def binary_code(Learner):
    for i in range(len(Learner.callbacks[0].outputs)):
        Learner.callbacks[0].outputs[i] = Learner.callbacks[0].outputs[i] >= 0.5

def similarity(ten1, ten2):
    return (ten1 == ten2).sum().float()/len(ten1)
