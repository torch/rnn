
These modules are DEPRECATED:
  * [FastLSTM](#rnn.FastLSTM) : an LSTM with optional support for batch normalization;
  * [Recurrent](#rnn.Recurrent) (DEPRECATED) : a generalized recurrent neural network container;
  * [LSTM](#rnn.LSTM) : a vanilla Long-Short Term Memory module (uses peephole connections);
  * [SeqLSTMP](#rnn.LSTM) : a vanilla Long-Short Term Memory module (uses peephole connections);

<a name='rnn.Recurrent'></a>
## Recurrent ##

DEPRECATED : use [Recurrence](#rnn.Recurrence) instead.

References :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer.

The `nn.Recurrent(start, input, feedback, [transfer, rho, merge])` constructor takes 6 arguments:
 * `start` : the size of the output (excluding the batch dimension), or a Module that will be inserted between the `input` Module and `transfer` module during the first step of the propagation. When `start` is a size (a number or `torch.LongTensor`), then this *start* Module will be initialized as `nn.Add(start)` (see Ref. A).
 * `input` : a Module that processes input Tensors (or Tables). Output must be of same size as `start` (or its output in the case of a `start` Module), and same size as the output of the `feedback` Module.
 * `feedback` : a Module that feedbacks the previous output Tensor (or Tables) up to the `merge` module.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input` and `feedback` Module before being forwarded through the `transfer` Module.
 * `transfer` : a non-linear Module used to process the output of the `merge` module, or in the case of the first step, the output of the `start` Module.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Due to the vanishing gradients effect, references A and B recommend `rho = 5` (or lower). Defaults to 99999.

An RNN is used to process a sequence of inputs.
Each step in the sequence should be propagated by its own `forward` (and `backward`),
one `input` (and `gradOutput`) at a time.
Each call to `forward` keeps a log of the intermediate states (the `input` and many `Module.outputs`)
and increments the `step` attribute by 1.
Method `backward` must be called in reverse order of the sequence of calls to `forward` in
order to backpropgate through time (BPTT). This reverse order is necessary
to return a `gradInput` for each call to `forward`.

The `step` attribute is only reset to 1 when a call to the `forget` method is made.
In which case, the Module is ready to process the next sequence (or batch thereof).
Note that the longer the sequence, the more memory that will be required to store all the
`output` and `gradInput` states (one for each time step).

To use this module with batches, we suggest using different
sequences of the same size within a batch and calling `updateParameters`
every `rho` steps and `forget` at the end of the sequence.

Note that calling the `evaluate` method turns off long-term memory;
the RNN will only remember the previous output. This allows the RNN
to handle long sequences without allocating any additional memory.


For a simple concise example of how to make use of this module, please consult the
[simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
training script.

<a name='rnn.Recurrent.Sequencer'></a>
### Decorate it with a Sequencer ###

Note that any `AbstractRecurrent` instance can be decorated with a [Sequencer](#rnn.Sequencer)
such that an entire sequence (a table) can be presented with a single `forward/backward` call.
This is actually the recommended approach as it allows RNNs to be stacked and makes the
rnn conform to the Module interface, i.e. each call to `forward` can be
followed by its own immediate call to `backward` as each `input` to the
model is an entire sequence, i.e. a table of tensors where each tensor represents
a time-step.

```lua
seq = nn.Sequencer(module)
```

The [simple-sequencer-network.lua](examples/simple-sequencer-network.lua) training script
is equivalent to the above mentionned [simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
script, except that it decorates the `rnn` with a `Sequencer` which takes
a table of `inputs` and `gradOutputs` (the sequence for that batch).
This lets the `Sequencer` handle the looping over the sequence.

You should only think about using the `AbstractRecurrent` modules without
a `Sequencer` if you intend to use it for real-time prediction.
Actually, you can even use an `AbstractRecurrent` instance decorated by a `Sequencer`
for real time prediction by calling `Sequencer:remember()` and presenting each
time-step `input` as `{input}`.

Other decorators can be used such as the [Repeater](#rnn.Repeater) or [RecurrentAttention](#rnn.RecurrentAttention).
The `Sequencer` is only the most common one.

<a name='rnn.LSTM'></a>
## LSTM ##
References :
 * A. [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/pdf/1303.5778v1.pdf)
 * B. [Long-Short Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
 * C. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
 * D. [nngraph LSTM implementation on github](https://github.com/wojzaremba/lstm)

This is an implementation of a vanilla Long-Short Term Memory module.
We used Ref. A's LSTM as a blueprint for this module as it was the most concise.
Yet it is also the vanilla LSTM described in Ref. C.

The `nn.LSTM(inputSize, outputSize, [rho])` constructor takes 3 arguments:
 * `inputSize` : a number specifying the size of the input;
 * `outputSize` : a number specifying the size of the output;
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999.

![LSTM](doc/image/LSTM.png)

The actual implementation corresponds to the following algorithm:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step,
`b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input,
`i[t]` is the input gate (eq. 1), `f[t]` is the forget gate (eq. 2),
`z[t]` is the input to the cell (which we call the hidden) (eq. 3),
`c[t]` is the cell (eq. 4), `o[t]` is the output gate (eq. 5),
and `h[t]` is the output of this module (eq. 6). Also note that the
weight matrices from cell to gate vectors are diagonal `W[c->s]`, where `s`
is `i`,`f`, or `o`.

As you can see, unlike [Recurrent](#rnn.Recurrent), this
implementation isn't generic enough that it can take arbitrary component Module
definitions at construction. However, the LSTM module can easily be adapted
through inheritance by overriding the different factory methods :
  * `buildGate` : builds generic gate that is used to implement the input, forget and output gates;
  * `buildInputGate` : builds the input gate (eq. 1). Currently calls `buildGate`;
  * `buildForgetGate` : builds the forget gate (eq. 2). Currently calls `buildGate`;
  * `buildHidden` : builds the hidden (eq. 3);
  * `buildCell` : builds the cell (eq. 4);
  * `buildOutputGate` : builds the output gate (eq. 5). Currently calls `buildGate`;
  * `buildModel` : builds the actual LSTM model which is used internally (eq. 6).



<a name='rnn.FastLSTM'></a>
## FastLSTM ##

DEPRECATED : use the much faster [RecLSTM](#rnn.RecLSTM) instead

A faster version of the [LSTM](#rnn.LSTM).
Basically, the input, forget and output gates, as well as the hidden state are computed at one fellswoop.

Note that `FastLSTM` does not use peephole connections between cell and gates. The algorithm from `LSTM` changes as follows:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
i.e. omitting the summands `W[c->i]c[t−1]` (eq. 1), `W[c->f]c[t−1]` (eq. 2), and `W[c->o]c[t]` (eq. 5).

### usenngraph ###
This is a static attribute of the `FastLSTM` class. The default value is `false`.
Setting `usenngraph = true` will force all new instantiated instances of `FastLSTM`
to use `nngraph`'s `nn.gModule` to build the internal `recurrentModule` which is
cloned for each time-step.

<a name='rnn.FastLSTM.bn'></a>
#### Recurrent Batch Normalization ####

This extends the `FastLSTM` class to enable faster convergence during training by zero-centering the input-to-hidden and hidden-to-hidden transformations.
It reduces the [internal covariate shift](https://arXiv.org/1502.03167v3) between time steps. It is an implementation of Cooijmans et. al.'s [Recurrent Batch Normalization](https://arxiv.org/1603.09025). The hidden-to-hidden transition of each LSTM cell is normalized according to
```lua
i[t] = σ(BN(W[x->i]x[t]) + BN(W[h->i]h[t−1]) + b[1->i])                      (1)
f[t] = σ(BN(W[x->f]x[t]) + BN(W[h->f]h[t−1]) + b[1->f])                      (2)
z[t] = tanh(BN(W[x->c]x[t]) + BN(W[h->c]h[t−1]) + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                                 (4)
o[t] = σ(BN(W[x->o]x[t]) + BN(W[h->o]h[t−1]) + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                        (6)
```
where the batch normalizing transform is:
```lua
  BN(h; gamma, beta) = beta + gamma *      hd - E(hd)
                                      ------------------
                                       sqrt(E(σ(hd) + eps))
```
where `hd` is a vector of (pre)activations to be normalized, `gamma`, and `beta` are model parameters that determine the mean and standard deviation of the normalized activation. `eps` is a regularization hyperparameter to keep the division numerically stable and `E(hd)` and `E(σ(hd))` are the estimates of the mean and variance in the mini-batch respectively. The authors recommend initializing `gamma` to a small value and found 0.1 to be the value that did not cause vanishing gradients. `beta`, the shift parameter, is `null` by default.

To turn on batch normalization during training, do:
```lua
nn.FastLSTM.bn = true
lstm = nn.FastLSTM(inputsize, outputsize, [rho, eps, momentum, affine]
```

where `momentum` is same as `gamma` in the equation above (defaults to 0.1), `eps` is defined above and `affine` is a boolean whose state determines if the learnable affine transform is turned off (`false`) or on (`true`, the default).


<a name='rnn.SeqLSTMP'></a>
## SeqLSTMP ##
References:
 * A. [LSTM RNN Architectures for Large Scale Acoustic Modeling](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf)
 * B. [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410v2.pdf)

```lua
lstmp = nn.SeqLSTMP(inputsize, hiddensize, outputsize)
```

The `SeqLSTMP` is a subclass of [SeqLSTM](#rnn.SeqLSTM).
It differs in that after computing the hidden state `h[t]` (eq. 6), it is
projected onto `r[t]` using a simple linear transform (eq. 7).
The computation of the gates also uses the previous such projection `r[t-1]` (eq. 1, 2, 3, 5).
This differs from `SeqLSTM` which uses `h[t-1]` instead of `r[t-1]`.

The computation of a time-step outlined in `SeqLSTM` is replaced with the following:
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
`SeqLSTMP` can be used with an `hiddensize >> outputsize` such that the effective size of the memory cells `c[t]`
and gates `i[t]`, `f[t]` and `o[t]` can be much larger than the actual input `x[t]` and output `r[t]`.
For fixed `inputsize` and `outputsize`, the `SeqLSTMP` will be able to remember much more information than the `SeqLSTM`.