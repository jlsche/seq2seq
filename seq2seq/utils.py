# coding: utf-8

import numpy as np

#from seq2seq.train import train
PAD = 0
EOS = 1

def batch(inputs, max_sequence_length=None):
    """
    Args: 
        inputs: list of sentences. (integer lists)

    Returns:
        time_major: input sentences transformed into time-major matrix.
                    (shape [max_time, batch_size] padded with 0s)
    """
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if not max_sequence_length:
        max_sequence_length = max(sequence_lengths)
    batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, ele in enumerate(seq):
            batch_major[i, j] = ele

    # [bach_size, max_time] => [max_time, batch_size]
    time_major = batch_major.swapaxes(0, 1)
    return time_major, sequence_lengths


def random_sequences(minlen=3, maxlen=8, batch_size=64):
    """ Yield batches of random integer sequences,
        sequence length will be in [minlen, maxlen].
    """

    if maxlen < minlen:
        raise ValueError('maxlen is less then minlen')

    def random_length():
        if minlen == maxlen:
            return minlen
        else:
            return np.random.randint(minlen, maxlen+1)

    def random_seq():
        sequence =  np.random.randint(low=2, high=10, size=random_length()).tolist()
        return sequence

    while True:
        yield [random_seq() for _ in range(batch_size)]


def feed(batches):
    """
    Args:
        batches: lists of sentences. (integer list)
    """
    if hasattr(batches, '__next__'): 
        _batch = next(batches)
    else:
        _batch = batches
    encoder_inputs_, _ = batch(_batch)
    decoder_targets_, _ = batch([(seq) + [EOS] for seq in _batch])
    decoder_inputs_, _ = batch([[EOS] + (seq) for seq in _batch])
    return {'encoder_inputs': encoder_inputs_, 
            'decoder_inputs': decoder_inputs_, 
            'decoder_targets': decoder_targets_}
