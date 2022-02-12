import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(
    X, y, item_dict, batch_size, maxlen, neg_examples, result_queue, SEED
):
    def sample():

        user = np.random.randint(0, len(X))
        while len(X[user]) <= 1:
            user = np.random.randint(0, len(X))

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        if neg_examples == 1:
            neg = np.zeros([maxlen], dtype=np.int32)
        else:
            neg = np.zeros([neg_examples, maxlen], dtype=np.int32)

        inputs, target = X[user][:maxlen], y[user][:maxlen]
        seq_len = max(len(inputs), len(target))
        inputs = [0] * (seq_len - len(inputs)) + inputs
        target = [0] * (seq_len - len(target)) + target
        # print(user, inputs, target)

        idx = maxlen - 1
        ts = set(target)
        for i, j in zip(reversed(inputs), reversed(target)):
            seq[idx] = i
            pos[idx] = j
            if j != 0:
                if neg_examples == 1:
                    neg[idx] = random_neq(1, len(item_dict) + 1, ts)
                else:
                    for ii in range(neg_examples):
                        neg[ii, idx] = random_neq(1, len(item_dict) + 1, ts)
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(
        self, X, y, item_dict, batch_size=64, maxlen=10, neg_examples=1, n_workers=1
    ):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        X,
                        y,
                        item_dict,
                        batch_size,
                        maxlen,
                        neg_examples,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
