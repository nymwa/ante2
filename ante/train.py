from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tunimi.vocabulary import Vocabulary
from ante.dataset import AnteDataset
from ante.sampler import Sampler
from soweli.soweli import Soweli

from ante.scheduler import WarmupScheduler
from ante.accumulator import Accumulator

from ante.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def make_dataset(vocab, dataset_path):
    with open(dataset_path) as f:
        sents = [x.strip().split() for x in f]
    sents = [[vocab.indices[token] for token in sent] for sent in sents]
    dataset = AnteDataset(sents, vocab)
    return dataset

def train(dataset_path,
        checkpoint_path,
        hidden_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        attention_dropout,
        dropout,
        max_tokens):

    vocab = Vocabulary()
    dataset = make_dataset(vocab, dataset_path)
    sampler = Sampler(dataset, max_tokens)
    loader = DataLoader(dataset, batch_sampler = sampler, collate_fn = dataset.collate)
    model = Soweli(len(vocab), hidden_size, nhead, hidden_size * 4,
            num_encoder_layers, num_decoder_layers,
            attention_dropout, dropout)
    model = model.cuda()

    print('#params (to train): {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('#params (total): {}'.format(sum(p.numel() for p in model.parameters())))

    optimizer = optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 0.01)
    scheduler = WarmupScheduler(optimizer, 1)
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.pad)

    clip_norm = 1.0
    num_steps = 0

    model.train()
    for epoch in range(100):
        accum = Accumulator(epoch, len(loader))
        for step, batch in enumerate(loader):
            batch.cuda()
            pred = model(batch)
            pred = pred.view(-1, pred.size(-1))

            loss = criterion(pred, batch.decoder_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()

            num_steps += 1
            lr = scheduler.get_last_lr()[0]
            accum.update(batch, loss, lr, grad)
            logger.info(accum.step_log())
        logger.info(accum.epoch_log(num_steps))

    torch.save(model.state_dict(), checkpoint_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='preprocessed.txt')
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-encoder-layers', type=int, default=3)
    parser.add_argument('--num-decoder-layers', type=int, default=3)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max-tokens', type=float, default=20000)
    args = parser.parse_args()

    train(args.dataset,
            args.checkpoint,
            args.hidden_size,
            args.nhead,
            args.num_encoder_layers,
            args.num_decoder_layers,
            args.attention_dropout,
            args.dropout,
            args.max_tokens)

