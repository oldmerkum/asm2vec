from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction

from asm2vec.model import Asm2Vec,Asm2VecMemento

import numpy as np

from tqdm import tqdm
from halo import Halo

import sys
import argparse
import time
from datetime import timedelta
import json

def peek_line(open_file):
    position = open_file.tell()
    line = open_file.readline()
    open_file.seek(position)
    return line

def next_line_is_instruction(open_file):
    next_line = peek_line(open_file)
    return next_line != '' and next_line[0] != ';' and next_line[0:2] != '::'

def next_line_is_block(open_file):
    return peek_line(open_file)[0:2] == '::'

def generate_block(asm_file):
    # assume next line is instruction
    block = BasicBlock()
    while next_line_is_instruction(asm_file):
        block.add_instruction(parse_instruction(asm_file.readline()))
    if next_line_is_block(asm_file):
        asm_file.readline()
        block.add_successor(generate_block(asm_file))
    return block

def generate_function(function_name, asm_file):
    # assume current_line is function name and next line is "::block"
    # consume the "::block" line
    asm_file.readline()
    return Function(generate_block(asm_file), function_name)

def test():
    print("setting up test")
    block1 = BasicBlock()
    block1.add_instruction(parse_instruction('mov eax, ebx'))
    block1.add_instruction(parse_instruction('jmp _loc'))

    block2 = BasicBlock()
    block2.add_instruction(parse_instruction('xor eax, eax'))
    block2.add_instruction(parse_instruction('ret'))

    block1.add_successor(block2)

    block3 = BasicBlock()
    block3.add_instruction(parse_instruction('sub eax, [ebp]'))

    block4 = BasicBlock()
    block4.add_instruction(parse_instruction('mov eax, ebx'))
    block4.add_instruction(parse_instruction('jmp _loc'))
    block4.add_instruction(parse_instruction('xor eax, eax'))
    block4.add_instruction(parse_instruction('ret'))

    f1 = Function(block1, 'some_func')
    f2 = Function(block3, 'another_func')

    # block4 is ignore here for clarity
    f3 = Function(block4, 'estimate_func')

    spinner = Halo(text="setting up model with functions", spinner='dots')
    model = Asm2Vec(d=200)
    train_repo = model.make_function_repo([f1, f2, f3])
    spinner.text = 'training test model'
    model.train(train_repo)
    spinner.stop()
    print("test training complete")
    return model

def build_manual_cfg(asm_filename):
    cfg = []
    with open(asm_filename, 'r') as asm_file:
        print(asm_file.readline())
        asm_file.seek(0)
        progress_bar = tqdm(total=(asm_file.read().count(';')))
        asm_file.seek(0)

        # process first line of asm file
        while peek_line(asm_file):
            # assume next line is ";<function_name>"
            function_name = asm_file.readline()[1:]
            cfg.append(generate_function(function_name, asm_file))
            progress_bar.update(1)
        progress_bar.close()
        print("Finished processing asm file")
    return cfg

def create_and_train_model(asm_filename):
    cfg = build_manual_cfg(asm_filename)
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
    cfg = build_manual_cfg(asm_filename)
    spinner = Halo(text="loading model", spinner='dots')
    spinner.start()
    model = load_saved_model(model_filename)
    spinner.text = 'training model'
    train_repo = model.make_function_repo(cfg)
    model.train(train_repo)
    spinner.stop()
    print('training complete')
    return model

def print_elapsed(start):
    elapsed = timedelta(seconds=(time.time() - start))
    print("time to train was {}".format(str(elapsed)))

def save_model(model, modelname):
    with open(modelname, 'w') as jsonfile:
        json.dump(model.memento().serialize(), jsonfile)
    print("saved model to {}".format(modelname))

def load_saved_model(modelfilename):
    model = Asm2Vec()
    with open(modelfilename, 'r') as jsonfile:
        loaded_data = Asm2VecMemento()
        loaded_data.populate(json.load(jsonfile))
        model.set_memento(loaded_data)
    return model

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
        model = test()
        json_filename = 'test.json'
    elif args.new_model:
        model = create_and_train_model(args.asm_filename)
        json_filename = args.asm_filename + ".json"
    else:
        model = load_and_train_model(args.model_file, args.asm_filename)
        json_filename = args.model_file

    print_elapsed(start)

    save_model(model, json_filename)
"""
    for tf in train_repo.funcs():
        print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))
"""

if __name__ == '__main__':
    sys.exit(main())