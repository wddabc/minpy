#!/usr/bin/python
# --------------------------------------- 
# File Name : deparser.py
# Creation Date : 08-03-2017
# Last Modified : Sun Mar 12 19:22:17 2017
# Created By : wdd 
# --------------------------------------- 
import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy import core
import numpy
import re


class Recurrent(object):
    def __init__(self, input_model, input_size=2,  # input dimension
                 hidden_size=64, name='Recurrent'):
        self.input_model = input_model
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size

    def add_param(self, name, shape):
        self.input_model.add_param(name=self.name + '_' + name, shape=shape)
        return self

    def params(self, name):
        return self.input_model.params[self.name + '_' + name]

    def step(self, x, h, *args, **kwargs):
        pass

    def output(self, h):
        pass

    def initial_state(self, batch_size=1):
        pass


class RNN(Recurrent):
    def __init__(self, input_model, input_size=2,  # input dimension
                 hidden_size=64, name='RNN'):
        super(RNN, self).__init__(input_model=input_model, input_size=input_size,
                                  hidden_size=hidden_size, name=name)
        self.add_param(name='Wx', shape=(input_size, hidden_size)) \
            .add_param(name='Wh', shape=(hidden_size, hidden_size)) \
            .add_param(name='b', shape=(hidden_size,))

    def step(self, x, h, *args, **kwargs):
        """
        Abstract step function for rnn
        :param x: current input, batched matrix
        :param h: previous output
        :return: output
        """
        return layers.rnn_step(x, h, self.params('Wx'),
                               self.params('Wh'), self.params('b'))

    def output(self, h):
        return h

    def initial_state(self, batch_size=1):
        return np.zeros((batch_size, self.hidden_size))


class LSTM(Recurrent):
    def __init__(self, input_model, input_size=2,  # input dimension
                 hidden_size=64, name='LSTM'):
        super(LSTM, self).__init__(input_model=input_model, input_size=input_size,
                                   hidden_size=hidden_size, name=name)
        self.add_param(name='Wx', shape=(input_size, 4 * hidden_size)) \
            .add_param(name='Wh', shape=(hidden_size, 4 * hidden_size)) \
            .add_param(name='b', shape=(4 * hidden_size,))

    def step(self, x, h, *args, **kwargs):
        """
        Abstract step function for rnn
        :param x: current input, batched matrix
        :param h: previous output, h[0] is the hidden output, h[1] is the memory cell
        :return: h[0] is the hidden output, h[1] is the memory cell
        """
        return layers.lstm_step(x, h[0], h[1], self.params('Wx'),
                                self.params('Wh'), self.params('b'))

    def initial_state(self, batch_size=1):
        return np.zeros((batch_size, self.hidden_size)), np.zeros((batch_size, self.hidden_size))

    def output(self, h):
        return h[0]


class StackRNN(object):
    def __init__(self, rnn, empty_embedding=None):
        """
        :param rnn: RNN cell
        :param empty_embedding: empty_embedding for initial layer
        """
        self.s = [(rnn.initial_state(), None)]
        self.rnn = rnn
        self.empty = empty_embedding

    def push(self, expr, extra=None):
        self.s.append((self.rnn.step(expr, self.s[-1][0]), extra))

    def pop(self):
        return self.s.pop()[1]  # return "extra" (i.e., whatever the caller wants or None)

    def top(self):
        # work around since inital_state.output() is None
        return self.rnn.output(self.s[-1][0]) if len(self.s) > 1 else self.empty

    def __len__(self):
        return len(self.s) - 1


class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)

    @classmethod
    def from_file(cls, vocab_fname):
        words = []
        with file(vocab_fname) as fh:
            for line in fh:
                line.strip()
                word, count = line.split()
                words.append(word)
        return Vocab.from_list(words)

    def size(self):
        return len(self.w2i.keys())


acts = ['SHIFT', 'REDUCE_L', 'REDUCE_R']
vocab_acts = Vocab.from_list(acts)
SHIFT = vocab_acts.w2i['SHIFT']
REDUCE_L = vocab_acts.w2i['REDUCE_L']
REDUCE_R = vocab_acts.w2i['REDUCE_R']
NUM_ACTIONS = vocab_acts.size()


def logsoftmax(x, valid_idx):
    x[np.array(valid_idx)] += 1e6
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))


class TransitionParser(ModelBase):
    def __init__(self, vocab, emb_size, hidden_size, num_act):
        self.vocab = vocab
        super(TransitionParser, self).__init__()
        # syntactic composition
        self.add_param(name='pW_comp', shape=(hidden_size, hidden_size * 2)) \
            .add_param(name='pb_comp', shape=(hidden_size, 1))
        # parser state to hidden
        self.add_param(name='pW_s2h', shape=(hidden_size, hidden_size * 2)) \
            .add_param(name='pb_s2h', shape=(hidden_size, 1))
        # hidden to action
        self.add_param(name='pW_act', shape=(num_act, hidden_size)) \
            .add_param(name='pb_act', shape=(num_act, 1))
        # layers, in-dim, out-dim, model
        self.buffRNN = LSTM(self, emb_size, hidden_size, name='buffRNN')
        self.stackRNN = LSTM(self, emb_size, hidden_size, name='stackRNN')
        self.add_param(name='empty_buffer_emb', shape=(1, hidden_size))
        self.add_param(name='wemb', shape=(vocab.size(), emb_size))

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, tokens, oracle_actions=None):
        def _valid_actions(stack, buffer):
            valid_actions = []
            if len(buffer) > 0:
                valid_actions += [SHIFT]
            if len(stack) >= 2:
                valid_actions += [REDUCE_L, REDUCE_R]
            return valid_actions

        if oracle_actions: oracle_actions = list(oracle_actions)
        buffer = StackRNN(self.buffRNN, self.params['empty_buffer_emb'])
        stack = StackRNN(self.stackRNN)

        # Put the parameters in the cg
        W_comp = self.params['pW_comp']  # syntactic composition
        b_comp = self.params['pb_comp']
        W_s2h = self.params['pW_s2h']  # state to hidden
        b_s2h = self.params['pb_s2h']
        W_act = self.params['pW_act']  # hidden to action
        b_act = self.params['pb_act']
        emb = self.params['wemb']

        # We will keep track of all the losses we accumulate during parsing.
        # If some decision is unambiguous because it's the only thing valid given
        # the parser state, we will not model it. We only model what is ambiguous.
        loss = 0.

        # push the tokens onto the buffer (tokens is in reverse order)
        for tok in tokens:
            # TODO: I remember numpy ndarray supports python built-in list indexing
            tok_embedding = emb[np.array([tok])]
            buffer.push(tok_embedding, (tok_embedding, self.vocab.i2w[tok]))
        while not (len(stack) == 1 and len(buffer) == 0):
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(stack, buffer)
            log_probs = None
            action = valid_actions[0]
            if len(valid_actions) > 1:
                p_t = np.transpose(np.concatenate([buffer.top(), stack.top()], axis=1))
                h = np.tanh(np.dot(W_s2h, p_t) + b_s2h)
                logits = np.dot(W_act, h) + b_act
                log_probs = logsoftmax(logits, valid_actions)
                if oracle_actions is None:
                    # Temporary work around by manually back-off to numpy https://github.com/dmlc/minpy/issues/15
                    action = numpy.argmax(map(lambda x: x[0], list(log_probs)))
            if oracle_actions is not None:
                action = oracle_actions.pop()
            if log_probs is not None:
                # append the action-specific loss
                # print action, log_probs[action], map(lambda x: x[0], list(log_probs))
                loss += log_probs[action]

            # execute the action to update the parser state
            if action == SHIFT:
                tok_embedding, token = buffer.pop()
                stack.push(tok_embedding, (tok_embedding, token))
            else:  # one of the REDUCE actions
                right = stack.pop()  # pop a stack state
                left = stack.pop()  # pop another stack state
                # figure out which is the head and which is the modifier
                head, modifier = (left, right) if action == REDUCE_R else (right, left)

                # compute composed representation
                head_rep, head_tok = head
                mod_rep, mod_tok = modifier
                composed_rep = np.tanh(
                    np.dot(W_comp, np.transpose(np.concatenate([head_rep, mod_rep], axis=1))) + b_comp)
                composed_rep = np.transpose(composed_rep)
                stack.push(composed_rep, (composed_rep, head_tok))
                if oracle_actions is None:
                    print('{0} --> {1}'.format(head_tok, mod_tok))

        # the head of the tree that remains at the top of the stack is the root
        if oracle_actions is None:
            head = stack.pop()[1]
            print('ROOT --> {0}'.format(head))
        return -loss


# format:
# John left . ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_R
def read_oracle(fname, vw, va):
    with file(fname) as fh:
        for line in fh:
            line = line.strip()
            ssent, sacts = re.split(r' \|\|\| ', line)
            sent = [vw.w2i[x] for x in ssent.split()]
            acts = [va.w2i[x] for x in sacts.split()]
            sent.reverse()
            acts.reverse()
            yield (sent, acts)


class MySolver(Solver):
    def __init__(self, model, **kwargs):
        from minpy.nn.io import NDArrayIter
        from examples.utils.data_utils import adding_problem_generator as data_gen
        x_train, y_train = data_gen(10000)
        x_test, y_test = data_gen(1000)

        train_dataiter = NDArrayIter(x_train,
                                     y_train,
                                     batch_size=100,
                                     shuffle=True)

        test_dataiter = NDArrayIter(x_test,
                                    y_test,
                                    batch_size=100,
                                    shuffle=False)

        super(MySolver, self).__init__(model, train_dataiter, test_dataiter, **kwargs)

    def train_on_batch(self, tokens, oracle_actions):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """

        # Compute loss and gradient
        def loss_func(*params):
            """
            Loss function calculate the loss
            """

            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            return self.model.parse(tokens, oracle_actions=oracle_actions)

        param_arrays = list(self.model.params.values())
        param_keys = list(self.model.params.keys())
        grad_and_loss_func = core.grad_and_loss(
            loss_func, argnum=range(len(param_arrays)))
        grad_arrays, loss = grad_and_loss_func(*param_arrays)
        print grad_arrays
        grads = dict(zip(param_keys, grad_arrays))

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
        return loss


def main():
    import argparse, random
    random.seed(0)
    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument('--vocab', default='s_lstm_deparser_data/vocab.txt', type=str, action='store', help='')
    _arg_parser.add_argument('--train', default='s_lstm_deparser_data/small-train.unk.txt', type=str, action='store', help='')
    _arg_parser.add_argument('--dev', default='s_lstm_deparser_data/small-dev.unk.txt', type=str, action='store', help='')
    _arg_parser.add_argument('--emb_size', default=64, type=int, action='store', help='')
    _arg_parser.add_argument('--rnn_size', default=64, type=int, action='store', help='')
    _args = _arg_parser.parse_args()

    # load training and dev data
    vocab_words = Vocab.from_file(_args.vocab)
    train = list(read_oracle(_args.train, vocab_words, vocab_acts))
    dev = list(read_oracle(_args.dev, vocab_words, vocab_acts))
    tp = TransitionParser(vocab_words, _args.emb_size, _args.rnn_size, NUM_ACTIONS)
    solver = MySolver(model=tp, update_rule='adam')
    solver.init()
    validation_losses = []

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    for epoch in range(5):
        random.shuffle(train)
        words = 0
        total_loss = 0.0
        for i, (s, a) in enumerate(train):
            instances_processed = float(i)
            # periodically report validation loss
            e = instances_processed / len(train)
            if instances_processed % 1000 == 0:
                dev_words = 0
                dev_loss = 0.0
                for (ds, da) in dev:
                    dev_words += len(ds)
                    dev_loss += np.mean(tp.parse(ds, da))
                    # print 'sent:', ds
                    # print 'oracle:', da
                    # exit()
                print('[validation] epoch {}: per-word loss: {}'.format(e, dev_loss / dev_words))
                validation_losses.append(dev_loss)

            # report training loss
            if instances_processed % 100 == 0 and words > 0:
                print('epoch {}: per-word loss: {}'.format(e, total_loss / words))
                words = 0
                total_loss = 0.0

            # here we do training
            total_loss += np.mean(
                solver.train_on_batch(s, a))  # returns None for 1-word sentencs (it's clear how to parse them)
            words += len(s)


if __name__ == '__main__':
    main()
