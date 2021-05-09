from pathlib import Path

def load_tokipona_vocabulary():
    path = Path(__file__).parent / 'vocabulary.txt'
    with open(path) as f:
        lst = [x.strip() for x in f]
    return lst

class Vocabulary(list):
    def __init__(self):
        self.pad, self.pad_id = '<pad>', 0
        self.eos, self.eos_id = '<eos>', 1
        self.unk, self.unk_id = '<unk>', 2
        token_list = [self.pad, self.eos, self.unk]

        self.number, self.number_id = '<number>', 3
        self.proper, self.proper_id = '<proper>', 4
        token_list += [self.number, self.proper]

        self.punctuation_list = list('!",.:?')
        self.punctuation_set = set(self.punctuation_list)
        token_list += self.punctuation_list

        token_list += load_tokipona_vocabulary()

        super().__init__(token_list)
        self.indices = {token: index for index, token in enumerate(self)}

