from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction

import asm2vec.model as model
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

def build_manual_cfg(asm_filename):
    cfg = []
    with open(asm_filename, 'r') as asm_file:
        print(asm_file.readline())
        asm_file.seek(0)
        progress_bar = tqdm(total=(asm_file.read().count(';')))
        asm_file.seek(0)

        # process first line of asm file
        asm_file.readline()
        while peek_line(asm_file):
            # assume next line is ";<function_name>"
            function_name = asm_file.readline()[1:]
            cfg.append(generate_function(function_name, asm_file))
            progress_bar.update(1)
        progress_bar.close()
        print("Finished processing asm file")
    return cfg

