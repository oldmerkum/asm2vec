from typing import *

import numpy as np

import asm2vec.asm
import asm2vec.repo

import asm2vec.internal.training
import asm2vec.internal.repr
import asm2vec.internal.util

import json

from halo import Halo

class Asm2VecMemento:
    def __init__(self):
        self.params: Optional[asm2vec.internal.training.Asm2VecParams] = None
        self.vocab: Optional[Dict[str, asm2vec.repo.Token]] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            'params': self.params.to_dict(),
            'vocab': asm2vec.repo.serialize_vocabulary(self.vocab)
        }

    def populate(self, rep: Dict[str, Any]) -> None:
        self.params = asm2vec.internal.training.Asm2VecParams()
        self.params.populate(rep['params'])
        self.vocab = asm2vec.repo.deserialize_vocabulary(rep['vocab'])


class Asm2Vec:
    def __init__(self, **kwargs):
        self._params = asm2vec.internal.training.Asm2VecParams(**kwargs)
        self._vocab = None

    def memento(self) -> Asm2VecMemento:
        memento = Asm2VecMemento()
        memento.params = self._params
        memento.vocab = self._vocab
        return memento

    def set_memento(self, memento: Asm2VecMemento) -> None:
        self._params = memento.params
        self._vocab = memento.vocab

    def make_function_repo(self, funcs: List[asm2vec.asm.Function]) -> asm2vec.repo.FunctionRepository:
        print('generating function repo')
        return asm2vec.internal.repr.make_function_repo(
            funcs, self._params.d, self._params.num_of_rnd_walks, self._params.jobs)

    def train(self, repo: asm2vec.repo.FunctionRepository) -> None:
        asm2vec.internal.training.train(repo, self._params)
        self._vocab = repo.vocab()

    def to_vec(self, f: asm2vec.asm.Function) -> np.ndarray:
        estimate_repo = asm2vec.internal.repr.make_estimate_repo(
            self._vocab, f, self._params.d, self._params.num_of_rnd_walks)
        vf = estimate_repo.funcs()[0]

        asm2vec.internal.training.estimate(vf, estimate_repo, self._params)

        return vf.v


def save_model(model, modelname):
    with open(modelname, 'w') as jsonfile:
        json.dump(model.memento().serialize(), jsonfile)
    print("saved model to {}".format(modelname))

def load_model(modelfilename):
    model = Asm2Vec()
    print('loading model...')
    with open(modelfilename, 'r') as jsonfile:
        loaded_data = Asm2VecMemento()
        loaded_data.populate(json.load(jsonfile))
        model.set_memento(loaded_data)

    return model

