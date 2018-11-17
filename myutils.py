from fastai.vision import *
from fastai import *

def run_folder(path, Learner):
    for i in (path).iterdir():
        Learner.predict(open_image(i))

def test():
        print('test')