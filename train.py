from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction

import asm2vec.model
from asm2vec.model import Asm2Vec,Asm2VecMemento

import customcfg
import tests.model_create_and_train_test as model_create_train_test

import numpy as np

from tqdm import tqdm
from halo import Halo

import sys
import argparse
import time
from datetime import timedelta
import json

def create_and_train_model(asm_filename):
    cfg = customcfg.build_manual_cfg(asm_filename)
    spinner = Halo(text="creating model", spinner='dots')
    spinner.start()
    model = Asm2Vec(d=200)
    spinner.text = 'training model'
    train_repo = model.make_function_repo(cfg)
    model.train(train_repo)
    spinner.stop()
    print("training complete")
    return model

def load_and_train_model(model_filename, asm_filename):
    cfg = customcfg.build_manual_cfg(asm_filename)
    spinner = Halo(text="loading model", spinner='dots')
    spinner.start()
    model = asm2vec.model.load_model(model_filename)
    spinner.text = 'training model'
    train_repo = model.make_function_repo(cfg)
    model.train(train_repo)
    spinner.stop()
    print('training complete')
    return model

def print_elapsed(start):
    elapsed = timedelta(seconds=(time.time() - start))
    print("time to train was {}".format(str(elapsed)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asm_filename", help="assembly file to train from")
    parser.add_argument("--test", help="run test on simple asm values",
                        action="store_true")
    parser.add_argument("--new-model", help="create new model to train",
                        action="store_true")
    parser.add_argument("--model-file", help="model to train")

    args = parser.parse_args()

    model = ''
    json_filename = ''

    start = time.time()
    if args.test:
        model = model_create_train_test.test()
        json_filename = 'test.json'
    elif args.new_model:
        model = create_and_train_model(args.asm_filename)
        json_filename = args.asm_filename + ".json"
    else:
        model = load_and_train_model(args.model_file, args.asm_filename)
        json_filename = args.model_file

    print_elapsed(start)

    asm2vec.model.save_model(model, json_filename)

if __name__ == '__main__':
    sys.exit(main())
