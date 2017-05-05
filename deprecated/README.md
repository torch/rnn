
These modules are DEPRECATED:
  * [FastLSTM](#rnn.FastLSTM) : an LSTM with optional support for batch normalization;
  * [LSTM](#rnn.LSTM) : a vanilla Long-Short Term Memory module (uses peephole connections);
  * [SeqLSTMP](#rnn.LSTM) : a vanilla Long-Short Term Memory module (uses peephole connections);
  * [GRU](#rnn.GRU) : a slower GRU than RecGRU;

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


<a name='rnn.GRU'></a>
## GRU ##

References :
 * A. [Learning Phrase Representations Using RNN Encoder-Decoder For Statistical Machine Translation.](http://arxiv.org/pdf/1406.1078.pdf)
 * B. [Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)
 * C. [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
 * D. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555)
 * E. [RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
 * F. [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

This is an implementation of Gated Recurrent Units module.

The `nn.GRU(inputSize, outputSize [,rho [,p [, mono]]])` constructor takes 3 arguments likewise `nn.LSTM` or 4 arguments for dropout:
 * `inputSize` : a number specifying the size of the input;
 * `outputSize` : a number specifying the size of the output;
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999;
 * `p` : dropout probability for inner connections of GRUs.

![GRU](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png)

The actual implementation corresponds to the following algorithm:
```lua
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step, `b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input and `s[t]` is the output of the module (eq. 4). Note that unlike the [RecLSTM](#rnn.RecLSTM), the GRU has no cells.

The GRU was benchmark on `PennTreeBank` dataset using [recurrent-language-model.lua](examples/recurrent-language-model.lua) script.
It slightly outperfomed [FastLSTM](https://github.com/torch/rnn/blob/master/deprecated/README.md#rnn.FastLSTM) (deprecated), however, since LSTMs have more parameters than GRUs,
the dataset larger than `PennTreeBank` might change the performance result.
Don't be too hasty to judge on which one is the better of the two (see Ref. C and D).

```
                Memory   examples/s
    FastLSTM      176M        16.5K
    GRU            92M        15.8K
```

__Memory__ is measured by the size of `dp.Experiment` save file. __examples/s__ is measured by the training speed at 1 epoch, so, it may have a disk IO bias.

![GRU-BENCHMARK](doc/image/gru-benchmark.png)

RNN dropout (see Ref. E and F) was benchmark on `PennTreeBank` dataset using `recurrent-language-model.lua` script, too. The details can be found in the script. In the benchmark, `GRU` utilizes a dropout after `LookupTable`, while `BGRU`, stands for Bayesian GRUs, uses dropouts on inner connections (naming as Ref. F), but not after `LookupTable`.

As Yarin Gal (Ref. F) mentioned, it is recommended that one may use `p = 0.25` for the first attempt.

![GRU-BENCHMARK](doc/image/bgru-benchmark.png)