import sys
from .normalizer import Normalizer
from .tokenizer import Tokenizer

def main():
    normalizer = Normalizer()
    tokenizer = Tokenizer()

    for x in sys.stdin:
        x = x.strip()
        x = normalizer(x)
        x = tokenizer(x, as_str=True)
        print(x)

