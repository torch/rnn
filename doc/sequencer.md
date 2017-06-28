<a name='rnn.sequencer'></a>
# Sequencer Modules #

Modules that `forward` entire sequences through an RNN :
 * [AbstractSequencer](sequencer.md#rnn.AbstractSequencer) : an abstract class inherited by Sequencer, Repeater, RecurrentAttention, etc.;
 * [Sequencer](sequencer.md#rnn.Sequencer) : applies an encapsulated module to all elements in an input sequence  (Tensor or Table);
 * [SeqLSTM](sequencer.md#rnn.SeqLSTM) : a faster version of `nn.Sequencer(nn.RecLSTM)` where the `input` and `output` are tensors;
 * [SeqGRU](sequencer.md#rnn.SeqGRU) : a faster version of `nn.Sequencer(nn.RecGRU)` where the `input` and `output` are tensors;
 * [BiSequencer](sequencer.md#rnn.BiSequencer) : used for implementing Bidirectional RNNs;
   * [SeqBLSTM](sequencer.md#rnn.SeqBLSTM) : bidirectional LSTM that uses two `SeqLSTMs` internally;
   * [SeqBGRU](sequencer.md#rnn.SeqBGRU) : bidirectional GRU that uses two `SeqGRUs` internally;
 * [Repeater](sequencer.md#rnn.Repeater) : repeatedly applies the same input to an `AbstractRecurrent` instance;
 * [RecurrentAttention](sequencer.md#rnn.RecurrentAttention) : a generalized attention model for [REINFORCE modules](reinforce.md#nn.Reinforce);

<a name='rnn.AbstractSequencer'></a>
## AbstractSequencer ##
This abstract class implements a light interface shared by
subclasses like : `Sequencer`, `Repeater`, `RecurrentAttention`, `BiSequencer` and so on.

<a name='rnn.AbstractSequencer.remember'></a>
### remember([mode]) ###
When `mode='neither'` (the default behavior of the class), the Sequencer will additionally call [forget](recurrent.md#rnn.AbstractRecurrent.forget) before each call to `forward`.
When `mode='both'` (the default when calling this function), the Sequencer will never call [forget](recurrent.md#rnn.AbstractRecurrent.forget).
In which case, it is up to the user to call `forget` between independent sequences.
This behavior is only applicable to decorated AbstractRecurrent `modules`.
Accepted values for argument `mode` are as follows :

 * 'eval' only affects evaluation (recommended for RNNs)
 * 'train' only affects training
 * 'neither' affects neither training nor evaluation (default behavior of the class)
 * 'both' affects both training and evaluation (recommended for LSTMs)

<a name='rnn.AbstractSequencer.hasMemory'></a>
### [bool] hasMemory()

Returns true if the instance has memory.
See [remember()](rnn.AbstractSequencer.remember) for details.

<a name='rnn.AbstractSequencer.setZeroMask'></a>
### setZeroMask(zeroMask)

Expects a `seqlen x batchsize` `zeroMask`.
The `zeroMask` is then passed to `seqlen` criterions by indexing `zeroMask[step]`.
When `zeroMask=false`, the zero-masking is disabled.

<a name='rnn.Sequencer'></a>
## Sequencer ##

The `nn.Sequencer(module)` constructor takes a single argument, `module`, which is the module
to be applied from left to right, on each element of the input sequence.

```lua
seq = nn.Sequencer(module)
```

The `Sequencer` is a kind of [decorator](http://en.wikipedia.org/wiki/Decorator_pattern)
used to abstract away the intricacies of `AbstractRecurrent` modules. While an `AbstractRecurrent` instance
requires that a sequence to be presented one input at a time, each with its own call to `forward` (and `backward`),
the `Sequencer` forwards an `input` sequence (a table) into an `output` sequence (a table of the same length).
It also takes care of calling `forget` on `AbstractRecurrent` instances.

The `Sequencer` inherits [AbstractSequencer](#rnn.AbstractSequencer)

### Input/Output Format

The `Sequencer` requires inputs and outputs to be of shape `seqlen x batchsize x featsize` :

 * `seqlen` is the number of time-steps that will be fed into the `Sequencer`.
 * `batchsize` is the number of examples in the batch. Each example is its own independent sequence.
 * `featsize` is the size of the remaining non-batch dimensions. So this could be `1` for language models, or `c x h x w` for convolutional models, etc.

![Hello Fuzzy](./image/hellofuzzy.png)

Above is an example input sequence for a character level language model.
It has `seqlen` is 5 which means that it contains sequences of 5 time-steps.
The openning `{` and closing `}` illustrate that the time-steps are elements of a Lua table, although
it also accepts full Tensors of shape `seqlen x batchsize x featsize`.
The `batchsize` is 2 as their are two independent sequences : `{ H, E, L, L, O }` and `{ F, U, Z, Z, Y, }`.
The `featsize` is 1 as their is only one feature dimension per character and each such character is of size 1.
So the input in this case is a table of `seqlen` time-steps where each time-step is represented by a `batchsize x featsize` Tensor.

![Sequence](./image/sequence.png)

Above is another example of a sequence (input or output).
It has a `seqlen` of 4 time-steps.
The `batchsize` is again 2 which means there are two sequences.
The `featsize` is 3 as each time-step of each sequence has 3 variables.
So each time-step (element of the table) is represented again as a tensor
of size `batchsize x featsize`.
Note that while in both examples the `featsize` encodes one dimension,
it could encode more.


### Example

For example, `rnn` : an instance of nn.AbstractRecurrent, can forward an `input` sequence one forward at a time:
```lua
input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
rnn:forward(input[1])
rnn:forward(input[2])
rnn:forward(input[3])
```

Equivalently, we can use a Sequencer to forward the entire `input` sequence at once:

```lua
seq = nn.Sequencer(rnn)
seq:forward(input)
```

We can also forward Tensors instead of Tables :

```lua
-- seqlen x batchsize x featsize
input = torch.randn(3,3,4)
seq:forward(input)
```

### Details

The `Sequencer` can also take non-recurrent Modules (i.e. non-AbstractRecurrent instances) and apply it to each
input to produce an output table of the same length.
This is especially useful for processing variable length sequences (tables).

Internally, the `Sequencer` expects the decorated `module` to be an
`AbstractRecurrent` instance. When this is not the case, the `module`
is automatically decorated with a [Recursor](recurrent.md#rnn.Recursor) module, which makes it
conform to the `AbstractRecurrent` interface.

Note : this is due a recent update (27 Oct 2015), as before this
`AbstractRecurrent` and and non-`AbstractRecurrent` instances needed to
be decorated by their own `Sequencer`. The recent update, which introduced the
`Recursor` decorator, allows a single `Sequencer` to wrap any type of module,
`AbstractRecurrent`, non-`AbstractRecurrent` or a composite structure of both types.
Nevertheless, existing code shouldn't be affected by the change.

For a concise example of its use, please consult the [simple-sequencer-network.lua](../examples/simple-sequencer-network.lua)
training script.

<a name='rnn.Sequencer.remember'></a>
### remember([mode]) ###
When `mode='neither'` (the default behavior of the class), the Sequencer will additionally call [forget](recurrent.md#rnn.AbstractRecurrent.forget) before each call to `forward`.
When `mode='both'` (the default when calling this function), the Sequencer will never call [forget](recurrent.md#rnn.AbstractRecurrent.forget).
In which case, it is up to the user to call `forget` between independent sequences.
This behavior is only applicable to decorated AbstractRecurrent `modules`.
Accepted values for argument `mode` are as follows :

 * 'eval' only affects evaluation (recommended for RNNs)
 * 'train' only affects training
 * 'neither' affects neither training nor evaluation (default behavior of the class)
 * 'both' affects both training and evaluation (recommended for LSTMs)

### forget() ###
Calls the decorated `AbstractRecurrent` module's `forget` method.

<a name='rnn.SeqLSTM'></a>
## SeqLSTM ##

This module is a faster version of `nn.Sequencer(nn.RecLSTM(inputsize, outputsize))` :

```lua
seqlstm = nn.SeqLSTM(inputsize, outputsize)
```

Each time-step is computed as follows (same as [RecLSTM](recurrent.md#rnn.RecLSTM)):

```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
```

A notable difference is that this module expects the `input` and `gradOutput` to
be tensors instead of tables. The default shape is `seqlen x batchsize x inputsize` for
the `input` and `seqlen x batchsize x outputsize` for the `output` :

```lua
input = torch.randn(seqlen, batchsize, inputsize)
gradOutput = torch.randn(seqlen, batchsize, outputsize)

output = seqlstm:forward(input)
gradInput = seqlstm:backward(input, gradOutput)
```

Note that if you prefer to transpose the first two dimension
(that is, `batchsize x seqlen` instead of the default `seqlen x batchsize`)
you can set `seqlstm.batchfirst = true` following initialization.

For variable length sequences, set `seqlstm.maskzero = true`.
This is equivalent to calling `RecLSTM:maskZero()` where the `RecLSTM` is wrapped by a `Sequencer`:
```lua
reclstm = nn.RecLSTM(inputsize, outputsize)
reclstm:maskZero(1)
seqlstm = nn.Sequencer(reclstm)
```

For `maskzero = true`, input sequences are expected to be seperated by tensor of zeros for a time step.


Like the `RecLSTM`, the `SeqLSTM` does not use peephole connections between cell and gates (see [RecLSTM](recurrent.md#rnn.RecLSTM) for details).

Like the `Sequencer`, the `SeqLSTM` provides a [remember](#rnn.Sequencer.remember) method.

Note that a `SeqLSTM` cannot replace `RecLSTM` in code that decorates it with a
`AbstractSequencer` or `Recursor` as this would be equivalent to `nn.Sequencer(nn.Sequencer(nn.RecLSTM))`.
You have been warned.

<a name='rnn.LSTMP'></a>
### LSTMP ###
References:
 * A. [LSTM RNN Architectures for Large Scale Acoustic Modeling](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf)
 * B. [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410v2.pdf)

```lua
lstmp = nn.SeqLSTM(inputsize, hiddensize, outputsize)
```

The `SeqLSTM` can implement an LSTM with a *projection* layer (LSTM*P*) when `hiddensize` and `outputsize` are provided.
An LSTMP differs from an LSTM in that after computing the hidden state `h[t]` (eq. 6), it is
projected onto `r[t]` using a simple linear transform (eq. 7).
The computation of the gates also uses the previous such projection `r[t-1]` (eq. 1, 2, 3, 5).
This differs from an LSTM which uses `h[t-1]` instead of `r[t-1]`.

The computation of a time-step outlined above for the LSTM is replaced with the following for an LSTMP:
```lua
i[t] = σ(W[x->i]x[t] + W[r->i]r[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[r->f]r[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]r[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[r->o]r[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
r[t] = W[h->r]h[t]                                                   (7)
```

The algorithm is outlined in ref. A and benchmarked with state of the art results on the Google billion words dataset in ref. B.
An LSTMP can be used with an `hiddensize >> outputsize` such that the effective size of the memory cells `c[t]`
and gates `i[t]`, `f[t]` and `o[t]` can be much larger than the actual input `x[t]` and output `r[t]`.
For fixed `inputsize` and `outputsize`, the LSTMP will be able to remember much more information than an LSTM.

<a name='rnn.SeqGRU'></a>
## SeqGRU ##

This module is a faster version of `nn.Sequencer(nn.GRU(inputsize, outputsize))` :

```lua
seqGRU = nn.SeqGRU(inputsize, outputsize)
```

Usage of SeqGRU differs from GRU in the same manner as SeqLSTM differs from LSTM. Therefore see [SeqLSTM](#rnn.SeqLSTM) for more details.

<a name='rnn.BiSequencer'></a>
## BiSequencer ##
Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing bidirectional RNNs like [SeqBLSTM](#rnn.SeqBLSTM) and [SeqBGRU](#rnn.SeqBGRU).

```lua
brnn = nn.BiSequencer(fwd, [bwd, merge])
```

The `input` to the module is a sequence tensor of size `seqlen x batchsize [x ...]`.
The `output` is a sequence of size `seqlen x batchsize [x ...]`.
`BiSequencer` applies a `fwd` RNN to each element in the sequence in
forward order and applies the `bwd` RNN in reverse order (from last element to first element).

The `fwd` and optional `bwd` RNN can be [AbstractRecurrent](recurrent.md#rnn.AbstractRecurrent) or [AbstractSequencer](#rnn.AbstractSequencer)
instances.

The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

For each step (in the original sequence), the outputs of both RNNs are merged together using
the `merge` module (defaults to `nn.CAddTable`).
This way, the outputs of both RNNs (in forward order) are summed.

Internally, the `BiSequencer` is implemented by decorating a structure of modules that makes
use of `Sequencers` for the `fwd` and `bwd` modules.

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note that when calling `BiSequencer:remember()`, only the `fwd` module can [remember()](#rnn.AbstractSequencer.remember).
The `bwd` module never remembers because it views the `input` in reverse order.

Also note that [`BiSequencer:setZeroMask(zeroMask)`](miscellaneous.md#rnn.MaskZero.setZeroMask)
corrently reverses the order of the `zeroMask` for the `bwd` RNN.


<a name='rnn.SeqBLSTM'></a>
## SeqBLSTM ##

```lua
blstm = nn.SeqBLSTM(inputsize, hiddensize, [outputsize])
```

A bi-directional RNN that uses `SeqLSTM`. Internally contains a `fwd` and `bwd` `SeqLSTM`.
Expects an input shape of `seqlen x batchsize x inputsize`.
For merging the outputs of the `fwd` and `bwd` LSTMs, this BLSTM uses `nn.CAddTable()`;
summing the outputs from eachoutput layer.

Example:
```
input = torch.rand(1, 2, 5)
blstm = nn.SeqBLSTM(5, 3)
print(blstm:forward(input))
```
Prints an output of a `1 x 2 x 3` tensor.

<a name='rnn.SeqBGRU'></a>
## SeqBGRU ##

```lua
blstm = nn.SeqBGRU(inputsize, outputsize)
```

A bi-directional RNN that uses `SeqGRU`. Internally contains a `fwd` and `bwd` `SeqGRU`.
Expects an input shape of `seqlen x batchsize x inputsize`.
For merging the outputs of the `fwd` and `bwd` LSTMs, this BLSTM uses `nn.CAddTable()`;
summing the outputs from eachoutput layer.

<a name='rnn.Repeater'></a>
## Repeater ##
This Module is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) similar to [Sequencer](#rnn.Sequencer).
It differs in that the sequence length is fixed before hand and the input is repeatedly forwarded
through the wrapped `module` to produce an output table of length `nStep`:
```lua
r = nn.Repeater(module, nStep)
```
Argument `module` should be an `AbstractRecurrent` instance.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same input.

<a name='rnn.RecurrentAttention'></a>
## RecurrentAttention ##
References :

  * A. [Recurrent Models of Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
  * B. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

This module can be used to implement the Recurrent Attention Model (RAM) presented in Ref. A :
```lua
ram = nn.RecurrentAttention(rnn, action, nStep, hiddenSize)
```

`rnn` is an [AbstractRecurrent](#rnn.AbstractRecurrent) instance.
Its input is `{x, z}` where `x` is the input to the `ram` and `z` is an
action sampled from the `action` module.
The output size of the `rnn` must be equal to `hiddenSize`.

`action` is a [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module)
that uses a [REINFORCE module](reinforce.md#nn.Reinforce) (ref. B) like
[ReinforceNormal](reinforce.md#nn.ReinforceNormal),
[ReinforceCategorical](reinforce.md#nn.ReinforceCategorical), or
[ReinforceBernoulli](reinforce.md#nn.ReinforceBernoulli)
to sample actions given the previous time-step's output of the `rnn`.
During the first time-step, the `action` module is fed with a Tensor of zeros of size `input:size(1) x hiddenSize`.
It is important to understand that the sampled actions do not receive gradients
backpropagated from the training criterion.
Instead, a reward is broadcast from a Reward Criterion like [VRClassReward](reinforce.md#nn.VRClassReward) Criterion to
the `action`'s REINFORCE module, which will backprogate graidents computed from the `output` samples
and the `reward`.
Therefore, the `action` module's outputs are only used internally, within the RecurrentAttention module.

`nStep` is the number of actions to sample, i.e. the number of elements in the `output` table.

`hiddenSize` is the output size of the `rnn`. This variable is necessary
to generate the zero Tensor to sample an action for the first step (see above).

A complete implementation of Ref. A is available [here](../examples/recurrent-visual-attention.lua).
