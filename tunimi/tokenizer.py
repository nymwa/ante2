import re
from .vocabulary import Vocabulary

class Tokenizer:
    def __init__(self):
        self.vocab = Vocabulary()
        self.proper_pattern = re.compile(r'^([AIUEO]|[KSNPML][aiueo]|[TJ][aueo]|W[aie])n?(([ksnpml][aiueo]|[tj][aueo]|w[aie])n?)*$')

    def convert(self, x):
        if x in self.vocab.indices:
            return self.vocab.indices[x]
        elif x.isdecimal():
            return self.vocab.number_id
        elif self.proper_pattern.match(x) and ('nm' not in x) and ('nn' not in x):
            return self.vocab.proper_id
        else:
            return self.vocab.unk_id

    def __call__(self, x, as_str=False):
        x = x.split()
        x = [self.convert(token) for token in x]
        if as_str:
            x = ' '.join([self.vocab[token] for token in x])
        return x

