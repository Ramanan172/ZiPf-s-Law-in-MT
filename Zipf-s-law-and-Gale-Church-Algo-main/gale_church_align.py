import math
from itertools import zip_longest
import numpy as np
from scipy.stats import norm

# Alignment costs: -100*log(p(x:y)/p(1:1))
bead_costs = {
    (1, 1): 0,
    (2, 1): 230,
    (1, 2): 230,
    (0, 1): 450,
    (1, 0): 450,
    (2, 2): 440
}

# Length cost parameters
mean_xy = 1
variance_xy = 6.8
LOG2 = math.log(2)


def length_cost(sx, sy):
    """ -100*log[p(|N(0, 1)|>delta)] """
    lx, ly = sum(sx), sum(sy)
    m = (lx + ly * mean_xy) / 2
    try:
        delta = (lx - ly * mean_xy) / np.sqrt(m * variance_xy)
    except ZeroDivisionError:
        return float('-inf')
    return -100 * (LOG2 + norm.logsf(abs(delta)))


def _align(x, y):
    m = {}
    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            if i == j == 0:
                m[i, j] = (0, 0, 0)
            else:
                m[i, j] = min((m[i - di, j - dj][0] +
                               length_cost(x[i - di:i], y[j - dj:j]) +
                               bead_cost,
                               di, dj)
                              for (di, dj), bead_cost in bead_costs.items()
                              if i - di >= 0 and j - dj >= 0)

    i, j = len(x), len(y)
    alignment = []
    while True:
        (c, di, dj) = m[i, j]
        if di == dj == 0:
            break
        alignment.append(((i - di, i), (j - dj, j)))
        i -= di
        j -= dj
    return alignment[::-1]  # Reverse the alignment for correct order


def char_length(sentence):
    """ Length of a sentence in characters """
    return sum(1 for c in sentence if c != ' ')


def align(sx, sy):
    """ Align two groups of sentences """
    cx = list(map(char_length, sx))
    cy = list(map(char_length, sy))
    for (i1, i2), (j1, j2) in _align(cx, cy):
        yield ' '.join(sx[i1:i2]), ' '.join(sy[j1:j2])


def read_blocks(f):
    block = []
    for line in f:
        if not line.strip():
            yield block
            block = []
        else:
            block.append(line.strip())
    if block:
        yield block


def main():
    # Paths to the text files
    corpus_x = 'Resources/sinhala_sentences.txt'
    corpus_y = 'Resources/english_sentences.txt'

    with open(corpus_x, encoding='utf-8') as fx, open(corpus_y, encoding='utf-8') as fy:
        for block_x, block_y in zip_longest(read_blocks(fx), read_blocks(fy), fillvalue=[]):
            for (sentence_x, sentence_y) in align(block_x, block_y):
                print('%s ||| %s' % (sentence_x, sentence_y))


if __name__ == '__main__':
    main()
