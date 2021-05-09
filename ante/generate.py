import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from ante.batch import Batch
from tunimi.normalizer import Normalizer
from tunimi.tokenizer import Tokenizer
from soweli.soweli import Soweli
from soweli.util import generate_square_subsequent_mask

class ScoredSentence:
    def __init__(self, log_probs, sent):
        self.log_probs = log_probs
        self.sent = sent

    def last(self):
        return self.sent[-1]

    def score(self):
        penalty = (len(self.log_probs) + 5) / 6
        penalty = penalty ** 0.6
        return sum(self.log_probs) / penalty

def make_constraints(sent, vocab):
    constraints = [0 for i in range(len(vocab))]
    for token in sent:
        if token != vocab.pad_id:
            constraints[token] += 1
    return constraints

class AnteScoredSentence(ScoredSentence):
    def __init__(self, log_probs, sent, constraints, vocab):
        super().__init__(log_probs, sent)
        self.vocab = vocab
        self.constraints = constraints

    def constrain(self, score):
        if all(x == 0 for x in self.constraints):
            self.constraints[self.vocab.eos_id] = 1

        for i in range(len(self.vocab)):
            if self.constraints[i] == 0:
                score[i] = - float('inf')
        return score

    def get_new_sent(self, log_prob, token):
        log_probs = self.log_probs + [log_prob]
        sent = self.sent + [token]
        constraints = self.constraints[:]
        constraints[token] -= 1
        return AnteScoredSentence(log_probs, sent, constraints, self.vocab)

def predict(model, vocab, mem, beam):
    decoder_inputs = [torch.tensor([vocab.eos_id] + sent.sent) for sent in beam]
    decoder_inputs = pad(decoder_inputs, padding_value = vocab.pad_id)
    decoder_inputs = decoder_inputs.cuda()
    attention_mask = generate_square_subsequent_mask(decoder_inputs.shape[0])
    attention_mask = attention_mask.cuda()
    mem = mem.repeat(1, decoder_inputs.shape[1], 1)

    with torch.no_grad():
        x = model.decode(decoder_inputs, mem,
                attention_mask = attention_mask)
    return x

def update_beam(old_beam, scores, width):
    new_beam = []
    for n in range(len(old_beam)):
        score = scores[-1, n]
        score = old_beam[n].constrain(score)
        score = torch.log_softmax(score, dim=-1)
        values, indices = score.topk(width)
        for value, index in zip(values, indices):
            if value != float('-inf'):
                sent = old_beam[n].get_new_sent(value.item(), index.item())
                new_beam.append(sent)
    new_beam.sort(key = lambda sent: -sent.score())
    new_beam = new_beam[:width]
    return new_beam

def split_beam(vocab, old_beam):
    new_beam = []
    ended = []
    for sent in old_beam:
        if sent.last() == vocab.eos_id:
            if all(num == 0 for num in sent.constraints):
                ended.append(sent)
        else:
            new_beam.append(sent)
    return new_beam, ended

def beam_search(model, vocab, width, input_sentence, max_len = 128):
    model.eval()
    mem = encode(model, vocab, input_sentence)
    beam = [AnteScoredSentence(log_probs = [], sent = [vocab.eos_id],
        constraints = make_constraints(input_sentence, vocab), vocab = vocab)]
    output = []
    for i in range(max_len):
        if len(beam) == 0:
            break
        scores = predict(model, vocab, mem, beam)
        beam = update_beam(beam, scores, width - len(output))
        beam, ended = split_beam(vocab, beam)
        output += ended
    output.sort(key = lambda sent:-sent.score())
    return output

def encode(model, vocab, x):
    encoder_inputs = torch.tensor([x]).cuda()
    encoder_inputs = pad(encoder_inputs, padding_value = vocab.pad_id)
    with torch.no_grad():
        h = model.encode(encoder_inputs)
    return h

def main():
    normalizer = Normalizer()
    tokenizer = Tokenizer()

    x = sys.stdin.readline().strip()
    x = normalizer(x)
    x = tokenizer(x)

    model = Soweli(len(tokenizer.vocab), 128, 4, 512, 3, 3, 0, 0)
    model.load_state_dict(torch.load('checkpoint.pt', map_location='cpu'))
    model = model.cuda()

    beam = beam_search(model, tokenizer.vocab, 12, x)
    for sent in beam:
        score = np.exp(sent.score())
        print(' '.join([tokenizer.vocab[n] for n in sent.sent]) + '\t' + str(score))

