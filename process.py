from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction

from asm2vec.model import Asm2Vec

import numpy as np

from tqdm import tqdm
from halo import Halo

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


functions = []

#TODO setup asking for filename
with open("file.elf.asm", 'r') as asm_file:
    print(asm_file.readline())
    asm_file.seek(0)
    progress_bar = tqdm(total=(asm_file.read().count(';')))
    asm_file.seek(0)

    # process first line of asm file
    while peek_line(asm_file):
        # assume next line is ";<function_name>"
        function_name = asm_file.readline()[1:]
        functions.append(generate_function(function_name, asm_file))
        progress_bar.update(1)
    progress_bar.close()
    print("Finished processing asm file")

spinner = Halo(text="setting up model with functions", spinner='dots')
spinner.start()
model = Asm2Vec(d=200)
spinner.text = 'training model'
train_repo = model.make_function_repo(functions)
model.train(train_repo)
spinner.stop()

print("training complete")

#TODO save the trained model

for tf in train_repo.funcs():
    print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))

(END)
