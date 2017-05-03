# Torch recurrent neural networks #

This is a recurrent neural network (RNN) library that extends Torch's nn.
You can use it to build RNNs, LSTMs, GRUs, BRNNs, BLSTMs, and so forth and so on.
This library includes documentation for the following objects:

Modules that consider successive calls to `forward` as different time-steps in a sequence :
 * [AbstractRecurrent](#rnn.AbstractRecurrent) : an abstract class inherited by `Recurrence` and `RecLSTM`;
 * [LookupRNN](#rnn.LookupRNN): implements a simple RNN where the input layer is a `LookupTable`;
 * [LinearRNN](#rnn.LinearRNN): implements a simple RNN where the input layer is a `Linear`;
 * [RecLSTM](#rnn.RecLSTM) : a faster LSTM (based on `SeqLSTM`) that doesn't use peephole connections;
 * [GRU](#rnn.GRU) : Gated Recurrent Units module;
 * [Recursor](#rnn.Recursor) : decorates a module to make it conform to the [AbstractRecurrent](#rnn.AbstractRecurrent) interface;
 * [Recurrence](#rnn.Recurrence) : decorates a module that outputs `output(t)` given `{input(t), output(t-1)}`;
 * [NormStabilizer](#rnn.NormStabilizer) : implements [norm-stabilization](http://arxiv.org/abs/1511.08400) criterion (add this module between RNNs);
 * [MuFuRu](#rnn.MuFuRu) : [Multi-function Recurrent Unit](https://arxiv.org/abs/1606.03002) module;

Modules that `forward` entire sequences through a decorated `AbstractRecurrent` instance :
 * [AbstractSequencer](#rnn.AbstractSequencer) : an abstract class inherited by Sequencer, Repeater, RecurrentAttention, etc.;
 * [Sequencer](#rnn.Sequencer) : applies an encapsulated module to all elements in an input sequence  (Tensor or Table);
 * [SeqLSTM](#rnn.SeqLSTM) : a faster version of `nn.Sequencer(nn.RecLSTM)` where the `input` and `output` are tensors;
 * [SeqGRU](#rnn.SeqGRU) : a very fast version of `nn.Sequencer(nn.GRU)` where the `input` and `output` are tensors;
 * [SeqBRNN](#rnn.SeqBRNN) : Bidirectional RNN based on SeqLSTM;
 * [BiSequencer](#rnn.BiSequencer) : used for implementing Bidirectional RNNs and LSTMs;
 * [BiSequencerLM](#rnn.BiSequencerLM) : used for implementing Bidirectional RNNs and LSTMs for language models;
 * [Repeater](#rnn.Repeater) : repeatedly applies the same input to an `AbstractRecurrent` instance;
 * [RecurrentAttention](#rnn.RecurrentAttention) : a generalized attention model for [REINFORCE modules](https://github.com/nicholas-leonard/dpnn#nn.Reinforce);

Miscellaneous modules and criterions :
 * [MaskZero](#rnn.MaskZero) : zeroes the `output` and `gradOutput` rows of the decorated module for commensurate `input` rows which are tensors of zeros;
 * [LookupTableMaskZero](#rnn.LookupTableMaskZero) : extends `nn.LookupTable` to support zero indexes for padding. Zero indexes are forwarded as tensors of zeros;
 * [MaskZeroCriterion](#rnn.MaskZeroCriterion) : zeros the `gradInput` and `loss` rows of the decorated criterion for commensurate `zeroMask` elements which are 1;
 * [SeqReverseSequence](#rnn.SeqReverseSequence) : reverses an input sequence on a specific dimension;
 * [VariableLength](#rnn.VariableLength): decorates a `Sequencer` to accept and produce a table of variable length inputs and outputs;

Criterions used for handling sequential inputs and targets :
 * [SequencerCriterion](#rnn.SequencerCriterion) : sequentially applies the same criterion to a sequence of inputs and targets (Tensor or Table).
 * [RepeaterCriterion](#rnn.RepeaterCriterion) : repeatedly applies the same criterion with the same target on a sequence.


This package also provides many useful features that aren't part of the main nn package.
These include [sharedClone](#nn.Module.sharedClone), which allows you to clone a module and share
parameters or gradParameters with the original module, without incuring any memory overhead.
We also redefined [type](#nn.Module.type) such that the type-cast preserves Tensor sharing within a structure of modules.

The package provides the following Modules:

 * [Decorator](#nn.Decorator) : abstract class to change the behaviour of an encapsulated module ;
 * [DontCast](#nn.DontCast) : prevent encapsulated module from being casted by `Module:type()` ;
 * [Serial](#nn.Serial) : decorate a module makes its serialized output more compact ;
 * [NaN](#nn.NaN) : decorate a module to detect the source of NaN errors ;
 * [Profile](#nn.Profile) : decorate a module to time its forwards and backwards passes ;
 * [Inception](#nn.Inception) : implements the Inception module of the GoogleLeNet article ;
 * [Collapse](#nn.Collapse) : just like `nn.View(-1)`;
 * [Convert](#nn.Convert) : convert between different tensor types or shapes;
 * [ZipTable](#nn.ZipTable) : zip a table of tables into a table of tables;
 * [ZipTableOneToMany](#nn.ZipTableOneToMany) : zip a table of element `el` and table of elements into a table of pairs of element `el` and table elements;
 * [CAddTensorTable](#nn.CAddTensorTable) : adds a tensor to a table of tensors of the same size;
 * [ReverseTable](#nn.ReverseTable) : reverse the order of elements in a table;
 * [PrintSize](#nn.PrintSize) : prints the size of inputs and gradOutputs (useful for debugging);
 * [Clip](#nn.Clip) : clips the inputs to a min and max value;
 * [Constant](#nn.Constant) : outputs a constant value given an input (which is ignored);
 * [SpatialUniformCrop](#nn.SpatialUniformCrop) : uniformly crops patches from a input;
 * [SpatialGlimpse](#nn.SpatialGlimpse) : takes a fovead glimpse of an image at a given location;
 * [WhiteNoise](#nn.WhiteNoise) : adds isotropic Gaussian noise to the signal when in training mode;
 * [OneHot](#nn.OneHot) : transforms a tensor of indices into [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding;
 * [Kmeans](#nn.Kmeans) : [Kmeans](https://en.wikipedia.org/wiki/K-means_clustering) clustering layer. Forward computes distances with respect to centroids and returns index of closest centroid. Centroids can be updated using gradient descent. Centroids could be initialized randomly or by using [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) algoirthm;
 * [SpatialRegionDropout](#nn.SpatialRegionDropout) : Randomly dropouts a region (top, bottom, leftmost, rightmost) of the input image. Works with batch and any number of channels;
 * [FireModule](#nn.FireModule) : FireModule as mentioned in the [SqueezeNet](http://arxiv.org/pdf/1602.07360v1.pdf);
 * [NCEModule](#nn.NCEModule) : optimized placeholder for a `Linear` + `SoftMax` using [noise-contrastive estimation](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf).
 * [SpatialFeatNormalization](#nn.SpatialFeatNormalization) : Module for widely used preprocessing step of mean zeroing and standardization for images.
 * [SpatialBinaryConvolution](#nn.SpatialBinaryConvolution) : Module for binary spatial convolution (Binary weights) as mentioned in [XNOR-Net](http://arxiv.org/pdf/1603.05279v2.pdf).
 * [SimpleColorTransform](#nn.SimpleColorTransform) : Module for adding independent random noise to input image channels.
 * [PCAColorTransform](#nn.PCAColorTransform) : Module for adding noise to input image using Principal Components Analysis.

The following modules and criterions can be used to implement the REINFORCE algorithm :

 * [Reinforce](#nn.Reinforce) : abstract class for REINFORCE modules;
 * [ReinforceBernoulli](#nn.ReinforceBernoulli) : samples from Bernoulli distribution;
 * [ReinforceNormal](#nn.ReinforceNormal) : samples from Normal distribution;
 * [ReinforceGamma](#nn.ReinforceGamma) : samples from Gamma distribution;
 * [ReinforceCategorical](#nn.ReinforceCategorical) : samples from Categorical (Multinomial with one sample) distribution;
 * [VRClassReward](#nn.VRClassReward) : criterion for variance-reduced classification-based reward;
 * [BinaryClassReward](#nn.BinaryClassReward) : criterion for variance-reduced binary classification reward (like `VRClassReward`, but for binary classes);

Additional differentiable criterions
 * [BinaryLogisticRegression](#nn.BLR) : criterion for binary logistic regression;
 * [SpatialBinaryLogisticRegression](#nn.SpatialBLR) : criterion for pixel wise binary logistic regression;
 * [NCECriterion](#nn.NCECriterion) : criterion exclusively used with [NCEModule](#nn.NCEModule).
 * [ModuleCriterion](#nn.ModuleCriterion) : adds an optional `inputModule` and `targetModule` before a decorated criterion;
 * [BinaryLogisticRegression](#nn.BLR) : criterion for binary logistic regression.
 * [SpatialBinaryLogisticRegression](#nn.SpatialBLR) : criterion for pixel wise binary logistic regression.


<a name='rnn.examples'></a>
## Examples ##

The following are example training scripts using this package :

  * [RNN/LSTM/GRU](examples/recurrent-language-model.lua) for Penn Tree Bank dataset;
  * [Noise Contrastive Estimate](examples/noise-contrastive-estimate.lua) for training multi-layer [SeqLSTM](#rnn.SeqLSTM) language models on the [Google Billion Words dataset](https://github.com/Element-Research/dataload#dl.loadGBW). The example uses [MaskZero](#rnn.MaskZero) to train independent variable length sequences using the [NCEModule](https://github.com/Element-Research/dpnn#nn.NCEModule) and [NCECriterion](https://github.com/Element-Research/dpnn#nn.NCECriterion). This script is our fastest yet boasting speeds of 20,000 words/second (on NVIDIA Titan X) with a 2-layer LSTM having 250 hidden units, a batchsize of 128 and sequence length of a 100. Note that you will need to have [Torch installed with Lua instead of LuaJIT](http://torch.ch/docs/getting-started.html#_);
  * [Recurrent Model for Visual Attention](examples/recurrent-visual-attention.lua) for the MNIST dataset;
  * [Encoder-Decoder LSTM](examples/encoder-decoder-coupling.lua) shows you how to couple encoder and decoder `LSTMs` for sequence-to-sequence networks;
  * [Simple Recurrent Network](examples/simple-recurrent-network.lua) shows a simple example for building and training a simple recurrent neural network;
  * [Simple Sequencer Network](examples/simple-sequencer-network.lua) is a version of the above script that uses the Sequencer to decorate the `rnn` instead;
  * [Sequence to One](examples/sequence-to-one.lua) demonstrates how to do many to one sequence learning as is the case for sentiment analysis;
  * [Multivariate Time Series](examples/recurrent-time-series.lua) demonstrates how train a simple RNN to do multi-variate time-series predication.

### External Resources

  * [rnn-benchmarks](https://github.com/glample/rnn-benchmarks) : benchmarks comparing Torch (using this library), Theano and TensorFlow.
  * [Harvard Jupyter Notebook Tutorial](http://nbviewer.jupyter.org/github/CS287/Lectures/blob/gh-pages/notebooks/ElementRNNTutorial.ipynb) : an in-depth tutorial for how to use the Element-Research rnn package by Harvard University;
  * [dpnn](https://github.com/Element-Research/dpnn) : this is a dependency of the __rnn__ package. It contains useful nn extensions, modules and criterions;
  * [dataload](https://github.com/Element-Research/dataload) : a collection of torch dataset loaders;
  * [RNN/LSTM/BRNN/BLSTM training script ](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua) for Penn Tree Bank or Google Billion Words datasets;
  * A brief (1 hours) overview of Torch7, which includes some details about the __rnn__ packages (at the end), is available via this [NVIDIA GTC Webinar video](http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4). In any case, this presentation gives a nice overview of Logistic Regression, Multi-Layer Perceptrons, Convolutional Neural Networks and Recurrent Neural Networks using Torch7;
  * [Sequence to Sequence mapping using encoder-decoder RNNs](https://github.com/rahul-iisc/seq2seq-mapping) : a complete training example using synthetic data.
  * [ConvLSTM](https://github.com/viorik/ConvLSTM) is a repository for training a [Spatio-temporal video autoencoder with differentiable memory](http://arxiv.org/abs/1511.06309).
  * An [time series example](https://github.com/rracinskij/rnntest01/blob/master/rnntest01.lua) for univariate timeseries prediction.
  * [Sagar Waghmare](https://github.com/sagarwaghmare69) wrote a nice [tutorial](tutorials/ladder.md) on how to use rnn with nngraph to reproduce the [Lateral Connections in Denoising Autoencoders Support Supervised Learning](http://arxiv.org/pdf/1504.08215.pdf).


## Citation ##

If you use __rnn__ in your work, we'd really appreciate it if you could cite the following paper:

Léonard, Nicholas, Sagar Waghmare, Yang Wang, and Jin-Hwa Kim. [rnn: Recurrent Library for Torch.](http://arxiv.org/abs/1511.07889) arXiv preprint arXiv:1511.07889 (2015).

Any significant contributor to the library will also get added as an author to the paper.
A [significant contributor](https://github.com/Element-Research/rnn/graphs/contributors)
is anyone who added at least 300 lines of code to the library.

## Troubleshooting ##

Most issues can be resolved by updating the various dependencies:
```bash
luarocks install torch
luarocks install nn
luarocks install dpnn
luarocks install torchx
```

If you are using CUDA :
```bash
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
```

And don't forget to update this package :
```bash
luarocks install rnn
```

If that doesn't fix it, open and issue on github.

<a name='rnn.AbstractRecurrent'></a>
## AbstractRecurrent ##
An abstract class inherited by [Recurrent](#rnn.Recurrent), [RecLSTM](#rnn.RecLSTM) and [GRU](#rnn.GRU).
The constructor takes a single argument :
```lua
rnn = nn.AbstractRecurrent([rho])
```
Argument `rho` is the maximum number of steps to backpropagate through time (BPTT).
Sub-classes can set this to a large number like 99999 (the default) if they want to backpropagate through
the entire sequence whatever its length. Setting lower values of rho are
useful when long sequences are forward propagated, but we only whish to
backpropagate through the last `rho` steps, which means that the remainder
of the sequence doesn't need to be stored (so no additional cost).

### [recurrentModule] getStepModule(step) ###
Returns a module for time-step `step`. This is used internally by sub-classes
to obtain copies of the internal `recurrentModule`. These copies share
`parameters` and `gradParameters` but each have their own `output`, `gradInput`
and any other intermediate states.

### setOutputStep(step) ###
This is a method reserved for internal use by [Recursor](#rnn.Recursor)
when doing backward propagation. It sets the object's `output` attribute
to point to the output at time-step `step`.
This method was introduced to solve a very annoying bug.

<a name='rnn.AbstractRecurrent.maskZero'></a>
### maskZero(nInputDim) ###
Decorates the internal `recurrentModule` with [MaskZero](#rnn.MaskZero).
The `output` Tensor (or table thereof) of the `recurrentModule`
will have each row (i.e. samples) zeroed when the commensurate row of the `input`
is a tensor of zeros.

The `nInputDim` argument must specify the number of non-batch dims
in the first Tensor of the `input`. In the case of an `input` table,
the first Tensor is the first one encountered when doing a depth-first search.

Calling this method makes it possible to pad sequences with different lengths in the same batch with zero vectors.

When a sample time-step is masked (i.e. `input` is a row of zeros), then
the hidden state is effectively reset (i.e. forgotten) for the next non-mask time-step.
In other words, it is possible seperate unrelated sequences with a masked element.

### trimZero(nInputDim) ###
Decorates the internal `recurrentModule` with [TrimZero](#rnn.TrimZero).

### [output] updateOutput(input) ###
Forward propagates the input for the current step. The outputs or intermediate
states of the previous steps are used recurrently. This is transparent to the
caller as the previous outputs and intermediate states are memorized. This
method also increments the `step` attribute by 1.

<a name='rnn.AbstractRecurrent.updateGradInput'></a>
### updateGradInput(input, gradOutput) ###
Like `backward`, this method should be called in the reverse order of
`forward` calls used to propagate a sequence. So for example :

```lua
rnn = nn.LSTM(10, 10) -- AbstractRecurrent instance
local outputs = {}
for i=1,nStep do -- forward propagate sequence
   outputs[i] = rnn:forward(inputs[i])
end

for i=nStep,1,-1 do -- backward propagate sequence in reverse order
   gradInputs[i] = rnn:backward(inputs[i], gradOutputs[i])
end

rnn:forget()
```

The reverse order implements backpropagation through time (BPTT).

### accGradParameters(input, gradOutput, scale) ###
Like `updateGradInput`, but for accumulating gradients w.r.t. parameters.

### recycle(offset) ###
This method goes hand in hand with `forget`. It is useful when the current
time-step is greater than `rho`, at which point it starts recycling
the oldest `recurrentModule` `sharedClones`,
such that they can be reused for storing the next step. This `offset`
is used for modules like `nn.Recurrent` that use a different module
for the first step. Default offset is 0.

<a name='rnn.AbstractRecurrent.forget'></a>
### forget(offset) ###
This method brings back all states to the start of the sequence buffers,
i.e. it forgets the current sequence. It also resets the `step` attribute to 1.
It is highly recommended to call `forget` after each parameter update.
Otherwise, the previous state will be used to activate the next, which
will often lead to instability. This is caused by the previous state being
the result of now changed parameters. It is also good practice to call
`forget` at the start of each new sequence.

<a name='rnn.AbstractRecurrent.maxBPTTstep'></a>
###  maxBPTTstep(rho) ###
This method sets the maximum number of time-steps for which to perform
backpropagation through time (BPTT). So say you set this to `rho = 3` time-steps,
feed-forward for 4 steps, and then backpropgate, only the last 3 steps will be
used for the backpropagation. If your AbstractRecurrent instance is wrapped
by a [Sequencer](#rnn.Sequencer), this will be handled auto-magically by the Sequencer.
Otherwise, setting this value to a large value (i.e. 9999999), is good for most, if not all, cases.

<a name='rnn.AbstractRecurrent.backwardOnline'></a>
### backwardOnline() ###
This method was deprecated Jan 6, 2016.
Since then, by default, `AbstractRecurrent` instances use the
backwardOnline behaviour.
See [updateGradInput](#rnn.AbstractRecurrent.updateGradInput) for details.

### training() ###
In training mode, the network remembers all previous `rho` (number of time-steps)
states. This is necessary for BPTT.

### evaluate() ###
During evaluation, since their is no need to perform BPTT at a later time,
only the previous step is remembered. This is very efficient memory-wise,
such that evaluation can be performed using potentially infinite-length
sequence.

<a name='rnn.Recurrent.Sequencer'></a>
<a name='rnn.AbstractRecurrent.Sequencer'></a>
### Decorate it with a Sequencer ###

Note that any `AbstractRecurrent` instance can be decorated with a [Sequencer](#rnn.Sequencer)
such that an entire sequence (a table or tensor) can be presented with a single `forward/backward` call.
This is actually the recommended approach as it allows RNNs to be stacked and makes the
RNN conform to the Module interface.
Each call to `forward` can be followed by its own immediate call to `backward` as each `input` to the
model is an entire sequence of size `seqlen x batchsize [x inputsize]`.

```lua
seq = nn.Sequencer(module)
```

The [simple-sequencer-network.lua](examples/simple-sequencer-network.lua) training script
is equivalent to the [simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
script.
The difference is that the former decorates the RNN with a `Sequencer` which takes
a table of `inputs` and `gradOutputs` (the sequence for that batch).
This lets the `Sequencer` handle the looping over the sequence.

You should only think about using the `AbstractRecurrent` modules without
a `Sequencer` if you intend to use it for real-time prediction.

Other decorators can be used such as the [Repeater](#rnn.Repeater) or [RecurrentAttention](#rnn.RecurrentAttention).
The `Sequencer` is only the most common one.

<a name='rnn.LookupRNN'></a>
## LookupRNN

References :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

This module subclasses the [Recurrence](#rnn.Recurrence) module to implement a simple RNN where the input layer is a
`LookupTable` module and the recurrent layer is a `Linear` module.
Note that to fully implement a Simple RNN, you need to add the output `Linear [+ SoftMax]` module after the `LookupRNN`.

The `nn.LookupRNN(nindex, outputsize, [transfer, merge])` constructor takes up to 4 arguments:
 * `nindex` : the number of embeddings in the `LookupTable(nindex, outputsize)` (that is, the *input layer*).
 * `outputsize` : the number of output units. This defines the size of the *recurrent layer* which is a `Linear(outputsize, outputsize)`.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `LookupTable` and `Linear` module before being forwarded through the `transfer` Module.  Defaults to `nn.CAddTable()`.
 * `transfer` : a non-linear modules used to process the output of the `merge` module. Defaults to `nn.Sigmoid()`.

The `LookupRNN` is essentially the following:

```lua
nn.Recurrence(
  nn.Sequential() -- input is {x[t], h[t-1]}
    :add(nn.ParallelTable()
      :add(nn.LookupTable(nindex, outputsize)) -- input layer
      :add(nn.Linear(outputsize, outputsize))) -- recurrent layer
    :add(merge)
    :add(transfer)
  , outputsize, 0)
```

An RNN is used to process a sequence of inputs.
As an `AbstractRecurrent` subclass, the `LookupRNN` propagates each step of a sequence by its own call to `forward` (and `backward`).
Each call to `LookupRNN.forward` keeps a log of the intermediate states (the `input` and many `Module.outputs`)
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
every `seqlen` steps and `forget` at the end of the sequence.

Note that calling the `evaluate` method turns off long-term memory;
the RNN will only remember the previous output. This allows the RNN
to handle long sequences without allocating any additional memory.

For a simple concise example of how to make use of this module, please consult the
[simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
training script.

<a name='rnn.LinearRNN'></a>
## LinearRNN

This module subclasses the [Recurrence](#rnn.Recurrence) module to implement a simple RNN where the input and the recurrent layer
are combined into a single `Linear` module.
Note that to fully implement the Simple RNN, you need to add the output `Linear [+ SoftMax]` module after the `LinearRNN`.

The `nn.LinearRNN(inputsize, outputsize, [transfer])` constructor takes up to 3 arguments:
 * `inputsize` : the number of input units;
 * `outputsize` : the number of output units.
 * `transfer` : a non-linear modules for activating the RNN. Defaults to `nn.Sigmoid()`.

The `LinearRNN` is essentially the following:

```lua
nn.Recurrence(
  nn.Sequential()
    :add(nn.JoinTable(1,1))
    :add(nn.Linear(inputsize+outputsize, outputsize))
    :add(transfer)
  , outputsize, 1)
```

Combining the input and recurrent layer into a single `Linear` module makes it quite efficient.

<a name='rnn.RecLSTM'></a>
## RecLSTM ##

References :
 * A. [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/pdf/1303.5778v1.pdf)
 * B. [Long-Short Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
 * C. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
 * D. [nngraph LSTM implementation on github](https://github.com/wojzaremba/lstm)

 ![LSTM](doc/image/LSTM.png)

Internally, `RecLSTM` uses a single module [StepLSTM](#rnn.StepLSTM), which is cloned (with shared parameters) for each time-step.
The speedup is obtained by computing every time-step using a single module.
This also makes the model memory efficient.

The algorithm for `RecLSTM` is as follows:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
```

Note that we recommend decorating the `RecLSTM` with a `Sequencer`
(refer to [this](#rnn.AbstractRecurrent.Sequencer) for details).
Also note that `RecLSTM` does not use peephole connections between cell and gates.

<a name='rnn.StepLSTM'></a>
### StepLSTM ###

`StepLSTM` is a step-wise module that can be used inside an `AbstractRecurrent` module to implement an LSTM.
For example, `StepLSTM` can be combined with [Recurrence](#rnn.Recurrence) (an `AbstractRecurrent` instance for create generic RNNs)
to create an LSTM:

```lua
local steplstm = nn.StepLSTM(inputsize, outputsize)
local stepmodule = nn.Sequential()
  :add(nn.FlattenTable())
  :add(steplstm)
local reclstm = nn.Sequential()
  :add(nn.Recurrence(stepmodule, {{outputsize}, {outputsize}}, 1, seqlen))
  :add(nn.SelectTable(1))
```

The above `reclstm` is functionally equivalent to a `RecLSTM`, although the latter is more efficient.

The `StepLSTM` thus efficiently implements a single LSTM time-step.
Its efficient because it doesn't use any internal modules; it calls BLAS directly.
`StepLSTM` is based on `SeqLSTM`.

The `input` to `StepLSTM` looks like:
```lua
{input[t], hidden[t-1], cell[t-1])}
```
where `t` indexes the time-step.
The `output` is:
```lua
{hidden[t], cell[t]}
```

### What is the difference between `SeqLSTM` and `RecLSTM`?

The `input` in `SeqLSTM:forward(input)` is a sequence of time-steps.
Whereas the `input` in `RecLSTM:forward(input)` is a single time-step.
`RecLSTM` is appropriate for real-time applications where `input[t]` depends on `output[t-1]`.
Use `RecLSTM` when the full sequence of `input` time-steps aren't known in advance.
For example, in a attention model, the next location to focus on depends on the previous recursion and location.

### LSTMP ###

Note that by calling `nn.RecLSTM(inputsize, hiddensize, outputsize)`
or `nn.StepLSTM(inputsize, hiddensize, outputsize)` (where both `hiddensize` and `outputsize` are numbers)
results in the creation of an [LSTMP](#rnn.LSTMP) instead of an LSTM.
An LSTMP is an LSTM with a projection layer.

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
 * `mono` : Monotonic sample for dropouts inside GRUs. Only needed in a `TrimZero` + `BGRU`(p>0) situation.

![GRU](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png)

The actual implementation corresponds to the following algorithm:
```lua
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step, `b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input and `s[t]` is the output of the module (eq. 4). Note that unlike the [LSTM](#rnn.LSTM), the GRU has no cells.

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

### SAdd

To implement `GRU`, a simple module is added, which cannot be possible to build only using `nn` modules.

```lua
module = nn.SAdd(addend, negate)
```
Applies a single scalar addition to the incoming data, i.e. y_i = x_i + b, then negate all components if `negate` is true. Which is used to implement `s[t] = (1-z[t])h[t] + z[t]s[t-1]` of `GRU` (see above Equation (4)).

```lua
nn.SAdd(-1, true)
```
Here, if the incoming data is `z[t]`, then the output becomes `-(z[t]-1)=1-z[t]`. Notice that `nn.Mul()` multiplies a scalar which is a learnable parameter.

<a name='rnn.MuFuRu'></a>
## MuFuRu ##

References :
 * A. [MuFuRU: The Multi-Function Recurrent Unit.](https://arxiv.org/abs/1606.03002)
 * B. [Tensorflow Implementation of the Multi-Function Recurrent Unit](https://github.com/dirkweissenborn/mufuru)

This is an implementation of the Multi-Function Recurrent Unit module.

The `nn.MuFuRu(inputSize, outputSize [,ops [,rho]])` constructor takes 2 required arguments, plus optional arguments:
 * `inputSize` : a number specifying the dimension of the input;
 * `outputSize` : a number specifying the dimension of the output;
 * `ops`: a table of strings, representing which composition operations should be used. The table can be any subset of `{'keep', 'replace', 'mul', 'diff', 'forget', 'sqrt_diff', 'max', 'min'}`. By default, all composition operations are enabled.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999;

The Multi-Function Recurrent Unit generalizes the GRU by allowing weightings of arbitrary composition operators to be learned. As in the GRU, the reset gate is computed based on the current input and previous hidden state, and used to compute a new feature vector:

```lua
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (1)
v[t] = tanh(W[x->v]x[t] + W[sr->v](s[t−1]r[t]) + b[1->v])  (2)
```

where `W[a->b]` denotes the weight matrix from activation `a` to `b`, `t` denotes the time step, `b[1->a]` is the bias for activation `a`, and `s[t-1]r[t]` is the element-wise multiplication of the two vectors.

Unlike in the GRU, rather than computing a single update gate (`z[t]` in [GRU](#rnn.GRU)), MuFuRU computes a weighting over an arbitrary number of composition operators.

A composition operator is any differentiable operator which takes two vectors of the same size, the previous hidden state, and a new feature vector, and returns a new vector representing the new hidden state. The GRU implicitly defines two such operations, `keep` and `replace`, defined as `keep(s[t-1], v[t]) = s[t-1]` and `replace(s[t-1], v[t]) = v[t]`.

[Ref. A](https://arxiv.org/abs/1606.03002) proposes 6 additional operators, which all operate element-wise:

* `mul(x,y) = x * y`
* `diff(x,y) = x - y`
* `forget(x,y) = 0`
* `sqrt_diff(x,y) = 0.25 * sqrt(|x - y|)`
* `max(x,y)`
* `min(x,y)`

The weightings of each operation are computed via a softmax from the current input and previous hidden state, similar to the update gate in the GRU. The produced hidden state is then the element-wise weighted sum of the output of each operation.
```lua

p^[t][j] = W[x->pj]x[t] + W[s->pj]s[t−1] + b[1->pj])         (3)
(p[t][1], ... p[t][J])  = softmax (p^[t][1], ..., p^[t][J])  (4)
s[t] = sum(p[t][j] * op[j](s[t-1], v[t]))                    (5)
```

where `p[t][j]` is the weightings for operation `j` at time step `t`, and `sum` in equation 5 is over all operators `J`.

<a name='rnn.Recursor'></a>
## Recursor ##

This module decorates a `module` to be used within an `AbstractSequencer` instance.
It does this by making the decorated module conform to the `AbstractRecurrent` interface,
which like the `RecLSTM` and `Recurrence` classes, this class inherits.

```lua
rec = nn.Recursor(module[, rho])
```

For each successive call to `updateOutput` (i.e. `forward`), this
decorator will create a `stepClone()` of the decorated `module`.
So for each time-step, it clones the `module`. Both the clone and
original share parameters and gradients w.r.t. parameters. However, for
modules that already conform to the `AbstractRecurrent` interface,
the clone and original module are one and the same (i.e. no clone).

Examples :

Let's assume I want to stack two LSTMs. I could use two sequencers :

```lua
lstm = nn.Sequential()
   :add(nn.Sequencer(nn.LSTM(100,100)))
   :add(nn.Sequencer(nn.LSTM(100,100)))
```

Using a `Recursor`, I make the same model with a single `Sequencer` :

```lua
lstm = nn.Sequencer(
   nn.Recursor(
      nn.Sequential()
         :add(nn.LSTM(100,100))
         :add(nn.LSTM(100,100))
      )
   )
```

Actually, the `Sequencer` will wrap any non-`AbstractRecurrent` module automatically,
so I could simplify this further to :

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.LSTM(100,100))
   )
```

I can also add a `Linear` between the two `LSTM`s. In this case,
a `Linear` will be cloned (and have its parameters shared) for each time-step,
while the `LSTM`s will do whatever cloning internally :

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.Linear(100,100))
      :add(nn.LSTM(100,100))
   )
```

`AbstractRecurrent` instances like `Recursor`, `RecLSTM` and `Recurrence` are
expcted to manage time-steps internally. Non-`AbstractRecurrent` instances
can be wrapped by a `Recursor` to have the same behavior.

Every call to `forward` on an `AbstractRecurrent` instance like `Recursor`
will increment the `self.step` attribute by 1, using a shared parameter clone
for each successive time-step (for a maximum of `rho` time-steps, which defaults to 9999999).
In this way, `backward` can be called in reverse order of the `forward` calls
to perform backpropagation through time (BPTT). Which is exactly what
[AbstractSequencer](#rnn.AbstractSequencer) instances do internally.
The `backward` call, which is actually divided into calls to `updateGradInput` and
`accGradParameters`, decrements by 1 the `self.udpateGradInputStep` and `self.accGradParametersStep`
respectively, starting at `self.step`.
Successive calls to `backward` will decrement these counters and use them to
backpropagate through the appropriate internall step-wise shared-parameter clones.

Anyway, in most cases, you will not have to deal with the `Recursor` object directly as
`AbstractSequencer` instances automatically decorate non-`AbstractRecurrent` instances
with a `Recursor` in their constructors.

For a concrete example of its use, please consult the [simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
training script for an example of its use.

<a name='rnn.Recurrence'></a>
## Recurrence ##

A extremely general container for implementing pretty much any type of recurrence.

```lua
rnn = nn.Recurrence(recurrentModule, outputSize, nInputDim, [rho])
```

`Recurrence` manages a single `recurrentModule`, which should
output a Tensor or table : `output(t)`
given an input table : `{input(t), output(t-1)}`.
Using a mix of `Recursor` (say, via `Sequencer`) with `Recurrence`, one can implement
pretty much any type of recurrent neural network, including LSTMs and RNNs.

For the first step, the `Recurrence` forwards a Tensor (or table thereof)
of zeros through the recurrent layer.
As such, `Recurrence` needs to know the `outputSize`, which is either a number or
`torch.LongStorage`, or table thereof. The batch dimension should be
excluded from the `outputSize`. Instead, the size of the batch dimension
(i.e. number of samples) will be extrapolated from the `input` using
the `nInputDim` argument. For example, say that our input is a Tensor of size
`4 x 3` where `4` is the number of samples, then `nInputDim` should be `1`.
As another example, if our input is a table of table [...] of tensors
where the first tensor (depth first) is the same as in the previous example,
then our `nInputDim` is also `1`.


As an example, let's use `Sequencer` and `Recurrence`
to build a Simple RNN for language modeling :

```lua
rho = 5
hiddenSize = 10
outputSize = 5 -- num classes
nIndex = 10000

-- recurrent module
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize))
      :add(nn.Linear(hiddenSize, hiddenSize)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

rnn = nn.Sequencer(
   nn.Sequential()
      :add(nn.Recurrence(rm, hiddenSize, 1))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
)
```

<a name='rnn.NormStabilizer'></a>
## NormStabilizer ##

Ref. A : [Regularizing RNNs by Stabilizing Activations](http://arxiv.org/abs/1511.08400)

This module implements the [norm-stabilization](http://arxiv.org/abs/1511.08400) criterion:

```lua
ns = nn.NormStabilizer([beta])
```

This module regularizes the hidden states of RNNs by minimizing the difference between the
L2-norms of consecutive steps. The cost function is defined as :
```
loss = beta * 1/T sum_t( ||h[t]|| - ||h[t-1]|| )^2
```
where `T` is the number of time-steps. Note that we do not divide the gradient by `T`
such that the chosen `beta` can scale to different sequence sizes without being changed.

The sole argument `beta` is defined in ref. A. Since we don't divide the gradients by
the number of time-steps, the default value of `beta=1` should be valid for most cases.

This module should be added between RNNs (or LSTMs or GRUs) to provide better regularization of the hidden states.
For example :
```lua
local stepmodule = nn.Sequential()
   :add(nn.RecLSTM(10,10))
   :add(nn.NormStabilizer())
   :add(nn.RecLSTM(10,10))
   :add(nn.NormStabilizer())
local rnn = nn.Sequencer(stepmodule)
```

To use it with `SeqLSTM` you can do something like this :
```lua
local rnn = nn.Sequential()
   :add(nn.SeqLSTM(10,10))
   :add(nn.Sequencer(nn.NormStabilizer()))
   :add(nn.SeqLSTM(10,10))
   :add(nn.Sequencer(nn.NormStabilizer()))
```

<a name='rnn.AbstractSequencer'></a>
## AbstractSequencer ##
This abstract class implements a light interface shared by
subclasses like : `Sequencer`, `Repeater`, `RecurrentAttention`, `BiSequencer` and so on.

<a name='rnn.Sequencer'></a>
## Sequencer ##

The `nn.Sequencer(module)` constructor takes a single argument, `module`, which is the module
to be applied from left to right, on each element of the input sequence.

```lua
seq = nn.Sequencer(module)
```

This Module is a kind of [decorator](http://en.wikipedia.org/wiki/Decorator_pattern)
used to abstract away the intricacies of `AbstractRecurrent` modules. While an `AbstractRecurrent` instance
requires that a sequence to be presented one input at a time, each with its own call to `forward` (and `backward`),
the `Sequencer` forwards an `input` sequence (a table) into an `output` sequence (a table of the same length).
It also takes care of calling `forget` on AbstractRecurrent instances.

### Input/Output Format

The `Sequencer` requires inputs and outputs to be of shape `seqlen x batchsize x featsize` :

 * `seqlen` is the number of time-steps that will be fed into the `Sequencer`.
 * `batchsize` is the number of examples in the batch. Each example is its own independent sequence.
 * `featsize` is the size of the remaining non-batch dimensions. So this could be `1` for language models, or `c x h x w` for convolutional models, etc.

![Hello Fuzzy](doc/image/hellofuzzy.png)

Above is an example input sequence for a character level language model.
It has `seqlen` is 5 which means that it contains sequences of 5 time-steps.
The openning `{` and closing `}` illustrate that the time-steps are elements of a Lua table, although
it also accepts full Tensors of shape `seqlen x batchsize x featsize`.
The `batchsize` is 2 as their are two independent sequences : `{ H, E, L, L, O }` and `{ F, U, Z, Z, Y, }`.
The `featsize` is 1 as their is only one feature dimension per character and each such character is of size 1.
So the input in this case is a table of `seqlen` time-steps where each time-step is represented by a `batchsize x featsize` Tensor.

![Sequence](doc/image/sequence.png)

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
is automatically decorated with a [Recursor](#rnn.Recursor) module, which makes it
conform to the `AbstractRecurrent` interface.

Note : this is due a recent update (27 Oct 2015), as before this
`AbstractRecurrent` and and non-`AbstractRecurrent` instances needed to
be decorated by their own `Sequencer`. The recent update, which introduced the
`Recursor` decorator, allows a single `Sequencer` to wrap any type of module,
`AbstractRecurrent`, non-`AbstractRecurrent` or a composite structure of both types.
Nevertheless, existing code shouldn't be affected by the change.

For a concise example of its use, please consult the [simple-sequencer-network.lua](examples/simple-sequencer-network.lua)
training script.

<a name='rnn.Sequencer.remember'></a>
### remember([mode]) ###
When `mode='neither'` (the default behavior of the class), the Sequencer will additionally call [forget](#nn.AbstractRecurrent.forget) before each call to `forward`.
When `mode='both'` (the default when calling this function), the Sequencer will never call [forget](#nn.AbstractRecurrent.forget).
In which case, it is up to the user to call `forget` between independent sequences.
This behavior is only applicable to decorated AbstractRecurrent `modules`.
Accepted values for argument `mode` are as follows :

 * 'eval' only affects evaluation (recommended for RNNs)
 * 'train' only affects training
 * 'neither' affects neither training nor evaluation (default behavior of the class)
 * 'both' affects both training and evaluation (recommended for LSTMs)

### forget() ###
Calls the decorated AbstractRecurrent module's `forget` method.

<a name='rnn.SeqLSTM'></a>
## SeqLSTM ##

This module is a faster version of `nn.Sequencer(nn.RecLSTM(inputsize, outputsize))` :

```lua
seqlstm = nn.SeqLSTM(inputsize, outputsize)
```

Each time-step is computed as follows (same as [RecLSTM](#rnn.RecLSTM)):

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


Like the `RecLSTM`, the `SeqLSTM` does not use peephole connections between cell and gates (see [RecLSTM](#rnn.RecLSTM) for details).

Like the `Sequencer`, the `SeqLSTM` provides a [remember](rnn.Sequencer.remember) method.

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

<a name='rnn.SeqBRNN'></a>
## SeqBRNN ##

```lua
brnn = nn.SeqBRNN(inputSize, outputSize, [batchFirst], [merge])
```

A bi-directional RNN that uses SeqLSTM. Internally contains a 'fwd' and 'bwd' module of SeqLSTM. Expects an input shape of ```seqlen x batchsize x inputsize```.
By setting [batchFirst] to true, the input shape can be ```batchsize x seqLen x inputsize```. Merge module defaults to CAddTable(), summing the outputs from each
output layer.

Example:
```
input = torch.rand(1, 1, 5)
brnn = nn.SeqBRNN(5, 5)
print(brnn:forward(input))
```
Prints an output of a 1x1x5 tensor.

<a name='rnn.BiSequencer'></a>
## BiSequencer ##
Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs.

```lua
brnn = nn.BiSequencer(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance) to each element in the sequence in
forward order and applies the `bwd` rnn in reverse order (from last element to first element).
The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`).
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Internally, the `BiSequencer` is implemented by decorating a structure of modules that makes
use of 3 Sequencers for the forward, backward and merge modules.

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note : make sure you call `brnn:forget()` after each call to `updateParameters()`.
Alternatively, one could call `brnn.bwdSeq:forget()` so that only `bwd` rnn forgets.
This is the minimum requirement, as it would not make sense for the `bwd` rnn to remember future sequences.


<a name='rnn.BiSequencerLM'></a>
## BiSequencerLM ##

Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs for Language Models (LM).

```lua
brnn = nn.BiSequencerLM(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance to the
first `N-1` elements in the sequence in forward order.
Applies the `bwd` rnn in reverse order to the last `N-1` elements (from second-to-last element to first element).
This is the main difference of this module with the [BiSequencer](#rnn.BiSequencer).
The latter cannot be used for language modeling because the `bwd` rnn would be trained to predict the input it had just be fed as input.

![BiDirectionalLM](doc/image/bidirectionallm.png)

The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

While the `fwd` rnn will output representations for the last `N-1` steps,
the `bwd` rnn will output representations for the first `N-1` steps.
The missing outputs for each rnn ( the first step for the `fwd`, the last step for the `bwd`)
will be filled with zero Tensors of the same size the commensure rnn's outputs.
This way they can be merged. If `nn.JoinTable` is used (the default), then the first
and last output elements will be padded with zeros for the missing `fwd` and `bwd` rnn outputs, respectively.

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`).
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note that LMs implemented with this module will not be classical LMs as they won't measure the
probability of a word given the previous words. Instead, they measure the probabiliy of a word
given the surrounding words, i.e. context. While for mathematical reasons you may not be able to use this to measure the
probability of a sequence of words (like a sentence),
you can still measure the pseudo-likeliness of such a sequence (see [this](http://arxiv.org/pdf/1504.01575.pdf) for a discussion).

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
that uses a [REINFORCE module](https://github.com/nicholas-leonard/dpnn#nn.Reinforce) (ref. B) like
[ReinforceNormal](https://github.com/nicholas-leonard/dpnn#nn.ReinforceNormal),
[ReinforceCategorical](https://github.com/nicholas-leonard/dpnn#nn.ReinforceCategorical), or
[ReinforceBernoulli](https://github.com/nicholas-leonard/dpnn#nn.ReinforceBernoulli)
to sample actions given the previous time-step's output of the `rnn`.
During the first time-step, the `action` module is fed with a Tensor of zeros of size `input:size(1) x hiddenSize`.
It is important to understand that the sampled actions do not receive gradients
backpropagated from the training criterion.
Instead, a reward is broadcast from a Reward Criterion like [VRClassReward](https://github.com/nicholas-leonard/dpnn#nn.VRClassReward) Criterion to
the `action`'s REINFORCE module, which will backprogate graidents computed from the `output` samples
and the `reward`.
Therefore, the `action` module's outputs are only used internally, within the RecurrentAttention module.

`nStep` is the number of actions to sample, i.e. the number of elements in the `output` table.

`hiddenSize` is the output size of the `rnn`. This variable is necessary
to generate the zero Tensor to sample an action for the first step (see above).

A complete implementation of Ref. A is available [here](examples/recurrent-visual-attention.lua).

<a name='rnn.MaskZero'></a>
## MaskZero ##
This module zeroes the `output` rows of the decorated module
for commensurate `input` rows which are tensors of zeros.

```lua
mz = nn.MaskZero(module, nInputDim)
```

The `output` Tensor (or table thereof) of the decorated `module`
will have each row (samples) zeroed when the commensurate row of the `input`
is a tensor of zeros.

The `nInputDim` argument must specify the number of non-batch dims
in the first Tensor of the `input`. In the case of an `input` table,
the first Tensor is the first one encountered when doing a depth-first search.

This decorator makes it possible to pad sequences with different lengths in the same batch with zero vectors.

Caveat: `MaskZero` not guarantee that the `output` and `gradInput` tensors of the internal modules
of the decorated `module` will be zeroed as well when the `input` is zero as well.
`MaskZero` only affects the immediate `gradInput` and `output` of the module that it encapsulates.
However, for most modules, the gradient update for that time-step will be zero because
backpropagating a gradient of zeros will typically yield zeros all the way to the input.
In this respect, modules to avoid in encapsulating inside a `MaskZero` are `AbsractRecurrent`
instances as the flow of gradients between different time-steps internally.
Instead, call the [AbstractRecurrent.maskZero](#rnn.AbstractRecurrent.maskZero) method
to encapsulate the internal `recurrentModule`.

<a name='rnn.TrimZero'></a>
## TrimZero ##

WARNING : only use this module if your input contains lots of zeros.
In almost all cases, [`MaskZero`](#rnn.MaskZero) will be faster, especially with CUDA.

Ref. A : [TrimZero: A Torch Recurrent Module for Efficient Natural Language Processing](https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIIS2016S_JHKim.pdf)

The usage is the same with `MaskZero`.

```lua
mz = nn.TrimZero(module, nInputDim)
```

The only difference from `MaskZero` is that it reduces computational costs by varying a batch size, if any, for the case that varying lengths are provided in the input.
Notice that when the lengths are consistent, `MaskZero` will be faster, because `TrimZero` has an operational cost.

In short, the result is the same with `MaskZero`'s, however, `TrimZero` is faster than `MaskZero` only when sentence lengths is costly vary.

In practice, e.g. language model, `TrimZero` is expected to be faster than `MaskZero` about 30%. (You can test with it using `test/test_trimzero.lua`.)

<a name='rnn.LookupTableMaskZero'></a>
## LookupTableMaskZero ##
This module extends `nn.LookupTable` to support zero indexes. Zero indexes are forwarded as zero tensors.

```lua
lt = nn.LookupTableMaskZero(nIndex, nOutput)
```

The `output` Tensor will have each row zeroed when the commensurate row of the `input` is a zero index.

This lookup table makes it possible to pad sequences with different lengths in the same batch with zero vectors.

<a name='rnn.MaskZeroCriterion'></a>
## MaskZeroCriterion ##

This criterion ignores samples (rows in the `input` and `target` tensors)
where the `zeroMask` ByteTensor passed to `MaskZeroCriterion:setZeroMask(zeroMask)` is 1.
This criterion only supports batch-mode.

```lua
batchsize = 3
zeroMask = torch.ByteTensor(batchsize):zero()
zeroMask[2] = 1 -- the 2nd sample in batch is ignored
mzc = nn.MaskZeroCriterion(criterion)
mzc:setZeroMask(zeroMask)
loss = mzc:forward(input, target)
gradInput = mzc:backward(input, target)
assert(gradInput[2]:sum() == 0)
```

In the above example, the second row of the `gradInput` Tensor is zero.
This is because the commensurate row in the `zeroMask` is a one.
The call to `forward` also disregards the second sample in measuring the `loss`.

This decorator makes it possible to pad sequences with different lengths in the same batch with zero vectors.

<a name='rnn.SeqReverseSequence'></a>
## SeqReverseSequence ##

```lua
reverseSeq = nn.SeqReverseSequence(dim)
```

Reverses an input tensor on a specified dimension. The reversal dimension can be no larger than three.

Example:
```lua
input = torch.Tensor({{1,2,3,4,5}, {6,7,8,9,10}})
reverseSeq = nn.SeqReverseSequence(1)
print(reverseSeq:forward(input))

Gives us an output of torch.Tensor({{6,7,8,9,10},{1,2,3,4,5}})
```

<a name='rnn.VariableLength'></a>
## VariableLength ##

```lua
vlrnn = nn.VariableLength(seqrnn, [lastOnly])
```

This module decorates a `seqrnn` to accept and produce a table of variable length inputs and outputs.
The `seqrnn` can be any module the accepts and produces a zero-masked sequence as input and output.
These include `Sequencer`, `SeqLSTM`, `SeqGRU`, and so on and so forth.

For example:
```lua
maxLength, hiddenSize, batchSize = 10, 4, 3
-- dummy variable length input
input = {}
for i=1,batchSize do
   -- each sample is a variable length sequence
   input[i] = torch.randn(torch.random(1,maxLength), hiddenSize)
end

-- create zero-masked LSTM (note calls to maskZero())
seqrnn = nn.Sequential()
   :add(nn.SeqLSTM(hiddenSize, hiddenSize):maskZero())
   :add(nn.Dropout(0.5))
   :add(nn.SeqLSTM(hiddenSize, hiddenSize):maskZero())

-- decorate with variable length module
vlrnn = nn.VariableLength(seqrnn)

output = vlrnn:forward(input)
print(output)
{
  1 : DoubleTensor - size: 7x4
  2 : DoubleTensor - size: 3x4
  3 : DoubleTensor - size: 2x4
}
```

By default `lastOnly` is false. When true, `vlrnn` only produces the last step of each variable-length sequence.
These last-steps are output as a tensor:

```lua
vlrnn.lastOnly = true
output = vlrnn:forward(input)
print(output)
-1.3430  0.1397 -0.1736  0.6332
-1.0903  0.2746 -0.3415 -0.2061
 0.7934  1.1306  0.8104  1.9069
[torch.DoubleTensor of size 3x4]
```

The module doesn't support CUDA.


<a name='rnn.SequencerCriterion'></a>
## SequencerCriterion ##

This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):

```lua
c = nn.SequencerCriterion(criterion, [sizeAverage])
```

Both the `input` and `target` are expected to be a sequence, either as a table or Tensor.
For each step in the sequence, the corresponding elements of the input and target
will be applied to the `criterion`.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful when used in conjunction with a [Sequencer](#rnn.Sequencer).

If `sizeAverage` is `true` (default is `false`), the `output` loss and `gradInput` is averaged over each time-step.

<a name='rnn.RepeaterCriterion'></a>
## RepeaterCriterion ##

This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):

```lua
c = nn.RepeaterCriterion(criterion)
```

The `input` is expected to be a sequence (table or Tensor). A single `target` is
repeatedly applied using the same `criterion` to each element in the `input` sequence.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same target.

<a name='nn.Module'></a>
## Module ##

The Module interface has been further extended with methods that facilitate
stochastic gradient descent like [updateGradParameters](#nn.Module.updageGradParameters) (i.e. momentum learning),
[weightDecay](#nn.Module.weightDecay), [maxParamNorm](#nn.Module.maxParamNorm) (for regularization), and so on.

<a name='nn.Module.dpnn_parameters'></a>
### Module.dpnn_parameters ###

A table that specifies the name of parameter attributes.
Defaults to `{'weight', 'bias'}`, which is a static variable (i.e. table exists in class namespace).
Sub-classes can define their own table statically.

<a name='nn.Module.dpnn_gradParameters'></a>
### Module.dpnn_gradParameters ###

A table that specifies the name of gradient w.r.t. parameter attributes.
Defaults to `{'gradWeight', 'gradBias'}`, which is a static variable (i.e. table exists in class namespace).
Sub-classes can define their own table statically.

<a name='nn.Module.type'></a>
### [self] Module:type(type_str) ###

This function converts all the parameters of a module to the given `type_str`.
The `type_str` can be one of the types defined for [torch.Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md)
like `torch.DoubleTensor`, `torch.FloatTensor` and `torch.CudaTensor`.
Unlike the [type method](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.type)
defined in [nn](https://github.com/torch/nn), this one was overriden to
maintain the sharing of [storage](https://github.com/torch/torch7/blob/master/doc/storage.md#storage)
among Tensors. This is especially useful when cloning modules share `parameters` and `gradParameters`.

<a name='nn.Module.sharedClone'></a>
### [clone] Module:sharedClone([shareParams, shareGradParams]) ###

Similar to [clone](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.clone).
Yet when `shareParams = true` (the default), the cloned module will share the parameters
with the original module.
Furthermore, when `shareGradParams = true` (the default), the clone module will share
the gradients w.r.t. parameters with the original module.
This is equivalent to :
```lua
clone = mlp:clone()
clone:share(mlp, 'weight', 'bias', 'gradWeight', 'gradBias')
```
yet it is much more efficient, especially for modules with lots of parameters, as these
Tensors aren't needlessly copied during the `clone`.
This is particularly useful for [Recurrent neural networks](https://github.com/Element-Research/rnn/blob/master/README.md)
which require efficient copies with shared parameters and gradient w.r.t. parameters for each time-step.

<a name='nn.Module.maxParamNorm'></a>
### Module:maxParamNorm([maxOutNorm, maxInNorm]) ###

This method implements a hard constraint on the upper bound of the norm of output and/or input neuron weights
[(Hinton et al. 2012, p. 2)](http://arxiv.org/pdf/1207.0580.pdf) .
In a weight matrix, this is a contraint on rows (`maxOutNorm`) and/or columns (`maxInNorm`), respectively.
Has a regularization effect analogous to [weightDecay](#nn.Module.weightDecay), but with easier to optimize hyper-parameters.
Assumes that parameters are arranged (`output dim x ... x input dim`).
Only affects parameters with more than one dimension.
The method should normally be called after [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters).
It uses the C/CUDA optimized [torch.renorm](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.renorm) function.
Hint : `maxOutNorm = 2` usually does the trick.

<a name='nn.Module.momentumGradParameters'></a>
### [momGradParams] Module:momentumGradParameters() ###

Returns a table of Tensors (`momGradParams`). For each element in the
table, a corresponding parameter (`params`) and gradient w.r.t. parameters
(`gradParams`) is returned by a call to [parameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.parameters).
This method is used internally by [updateGradParameters](#nn.Module.updateGradParameters).

<a name='nn.Module.updateGradParameters'></a>
### Module:updateGradParameters(momFactor [, momDamp, momNesterov]) ###

Applies classic momentum or Nesterov momentum [(Sutskever, Martens et al, 2013)](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf) to parameter gradients.
Each parameter Tensor (`params`) has a corresponding Tensor of the same size for gradients w.r.t. parameters (`gradParams`).
When using momentum learning, another Tensor is added for each parameter Tensor (`momGradParams`).
This method should be called before [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters)
as it affects the gradients w.r.t. parameters.

Classic momentum is computed as follows :

```lua
momGradParams = momFactor*momGradParams + (1-momDamp)*gradParams
gradParams = momGradParams
```

where `momDamp` has a default value of `momFactor`.

Nesterov momentum (`momNesterov = true`) is computed as follows (the first line is the same as classic momentum):

```lua
momGradParams = momFactor*momGradParams + (1-momDamp)*gradParams
gradParams = gradParams + momFactor*momGradParams
```
The default is to use classic momentum (`momNesterov = false`).

<a name='nn.Module.weightDecay'></a>
### Module:weightDecay(wdFactor [, wdMinDim]) ###

Decays the weight of the parameterized models.
Implements an L2 norm loss on parameters with dimensions greater or equal to `wdMinDim` (default is 2).
The resulting gradients are stored into the corresponding gradients w.r.t. parameters.
Such that this method should be called before [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters).

<a name='nn.Module.gradParamClip'></a>
### Module:gradParamClip(cutoffNorm [, moduleLocal]) ###

Implements a contrainst on the norm of gradients w.r.t. parameters [(Pascanu et al. 2012)](http://arxiv.org/pdf/1211.5063.pdf).
When `moduleLocal = false` (the default), the norm is calculated globally to Module for which this is called.
So if you call it on an MLP, the norm is computed on the concatenation of all parameter Tensors.
When `moduleLocal = true`, the norm constraint is applied
to the norm of all parameters in each component (non-container) module.
This method is useful to prevent the exploding gradient in
[Recurrent neural networks](https://github.com/Element-Research/rnn/blob/master/README.md).

<a name='nn.Module.reinforce'></a>
### Module:reinforce(reward) ###

This method is used by Criterions that implement the REINFORCE algorithm like [VRClassReward](#nn.VRClassReward).
While vanilla backpropagation (gradient descent using the chain rule),
REINFORCE Criterions broadcast a `reward` to all REINFORCE modules between the `forward` and the `backward`.
In this way, when the following call to `backward` reaches the REINFORCE modules,
these will compute a `gradInput` using the broadcasted `reward`.
The `reward` is broadcast to all REINFORCE modules contained
within `model` by calling `model:reinforce(reward)`.
Note that the `reward` should be a 1D tensor of size `batchSize`,
i.e. each example in a batch has its own scalar reward.

Refer to [this example](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)
for a complete training script making use of the REINFORCE interface.

<a name='nn.Decorator'></a>
## Decorator ##

```lua
dmodule = nn.Decorator(module)
```

This module is an abstract class used to decorate a `module`. This means
that method calls to `dmodule` will call the same method on the encapsulated
`module`, and return its results.

<a name='nn.DontCast'></a>
## DontCast ##

```lua
dmodule = nn.DontCast(module)
```

This module is a decorator. Use it to decorate a module that you don't
want to be cast when the `type()` method is called.

```lua
module = nn.DontCast(nn.Linear(3,4):float())
module:double()
th> print(module:forward(torch.FloatTensor{1,2,3}))
 1.0927
-1.9380
-1.8158
-0.0805
[torch.FloatTensor of size 4]
```

<a name='nn.Serial'></a>
## Serial ##

```lua
dmodule = nn.Serial(module, [tensortype])
dmodule:[light,medium,heavy]Serial()
```

This module is a decorator that can be used to control the serialization/deserialization
behavior of the encapsulated module. Basically, making the resulting string or
file heavy (the default), medium or light in terms of size.

Furthermore, when specified, the `tensortype` attribute (e.g *torch.FloatTensor*, *torch.DoubleTensor* and so on.),
determines what type the module will be cast to during serialization.
Note that this will also be the type of the deserialized object.
The default serialization `tensortype` is `nil`, i.e. the module is serialized as is.

The `heavySerial()` has the serialization process serialize every attribute in the module graph,
which is the default behavior of nn.

The `mediumSerial()` has the serialization process serialize
everything except the attributes specified in each module's `dpnn_mediumEmpty`
table, which has a default value of `{'output', 'gradInput', 'momGradParams', 'dpnn_input'}`.
During serialization, whether they be tables or Tensors, these attributes are emptied (no storage).
Some modules overwrite the default `Module.dpnn_mediumEmpty` static attribute with their own.

The `lightSerial()` has the serialization process empty
everything a call to `mediumSerial(type)` would (so it uses `dpnn_mediumEmpty`).
But also empties all the parameter gradients specified by the
attribute `dpnn_gradParameters`, which defaults to `{gradWeight, gradBias}`.

We recomment using `mediumSerial()` for training, and `lightSerial()` for
production (feed-forward-only models).

<a name='nn.NaN'></a>
## NaN ##

```lua
dmodule = nn.NaN(module, [id])
```

The `NaN` module asserts that the `output` and `gradInput` of the decorated `module` do not contain NaNs.
This is useful for locating the source of those pesky NaN errors.
The `id` defaults to automatically incremented values of `1,2,3,...`.

For example :

```lua
linear = nn.Linear(3,4)
mlp = nn.Sequential()
mlp:add(nn.NaN(nn.Identity()))
mlp:add(nn.NaN(linear))
mlp:add(nn.NaN(nn.Linear(4,2)))
print(mlp)
```

As you can see the `NaN` layers are have unique ids :

```lua
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.NaN(1) @ nn.Identity
  (2): nn.NaN(2) @ nn.Linear(3 -> 4)
  (3): nn.NaN(3) @ nn.Linear(4 -> 2)
}
```

And if we fill the `bias` of the linear module with NaNs and call `forward`:

```lua
nan = math.log(math.log(0)) -- this is a nan value
linear.bias:fill(nan)
mlp:forward(torch.randn(2,3))
```

We get a nice error message:
```lua
/usr/local/share/lua/5.1/dpnn/NaN.lua:39: NaN found in parameters of module :
nn.NaN(2) @ nn.Linear(3 -> 4)
```

For a quick one-liner to catch NaNs anywhere inside a model (for example, a `nn.Sequential` or any other `nn.Container`), we can use this with the `nn.Module.replace` function:
```lua
model:replace(function(module) return nn.NaN(module) end)
```

<a name='nn.Profile'></a>
## Profile ##

```lua
dmodule = nn.Profile(module, [print_interval, [name] ])
```

The `Profile` module times each forward and backward pass of the decorated `module`. It prints this information after `print_interval` passes, which is `100` by default. For timing multiple modules, the `name` argument allows this information to be printed accompanied by a name, which by default is the type of the decorated `module`.

This is useful for profiling new modules you develop, and tracking down bottlenecks in the speed of a network.

The timer and print statement can add a small amount of overhead to the overall speed.

As an example:

```lua
mlp = nn.Sequential()
mlp:add(nn.Identity())
mlp:add(nn.Linear(1000,1000))
mlp:add(nn.Tanh())
mlp:replace(function(module) return nn.Profile(module, 1000) end)
inp = torch.randn(1000)
gradOutput = torch.randn(1000)
for i=1,1000 do
   mlp:forward(inp)
   mlp:backward(inp, gradOutput)
end
```

results in the following profile information:

```
nn.Identity took 0.026 seconds for 1000 forward passes
nn.Linear took 0.119 seconds for 1000 forward passes
nn.Tanh took 0.061 seconds for 1000 forward passes
nn.Tanh took 0.032 seconds for 1000 backward passes
nn.Linear took 0.161 seconds for 1000 backward passes
nn.Identity took 0.026 seconds for 1000 backward passes
```

It's good practice to profile modules after a single forwards and backwards pass, since the initial pass often has to allocate memory. Thus, in the example above, you would run another 1000 forwards and backwards passes to time the modules in their normal mode of operation:

```
for i=1,1000 do
   mlp:forward(inp)
   mlp:backward(inp, gradOutput)
end
```

<a name='nn.Inception'></a>
## Inception ##
References :

  * A. [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)
  * B. [GoogleLeNet](http://image-net.org/challenges/LSVRC/2014/slides/GoogLeNet.pptx)

```lua
module = nn.Inception(config)
```

This module uses `n`+2 parallel "columns".
The original paper uses 2+2 where the first two are (but there could be more than two):

  * 1x1 conv (reduce) -> relu -> 5x5 conv -> relu
  * 1x1 conv (reduce) -> relu -> 3x3 conv -> relu

and where the other two are :

  * 3x3 maxpool -> 1x1 conv (reduce/project) -> relu
  * 1x1 conv (reduce) -> relu.

This module allows the first group of columns to be of any
number while the last group consist of exactly two columns.
The 1x1 convoluations are used to reduce the number of input channels
(or filters) such that the capacity of the network doesn't explode.
We refer to these here has *reduce*.
Since each column seems to have one and only one reduce, their initial
configuration options are specified in lists of n+2 elements.

The sole argument `config` is a table taking the following key-values :

  * Required Arguments :
   * `inputSize` : number of input channels or colors, e.g. 3;
   * `outputSize` : numbers of filters in the non-1x1 convolution kernel sizes, e.g. `{32,48}`
   * `reduceSize` : numbers of filters in the 1x1 convolutions (reduction) used in each column, e.g. `{48,64,32,32}`. The last 2 are used respectively for the max pooling (projection) column (the last column in the paper) and the column that has nothing but a 1x1 conv (the first column in the paper). This table should have two elements more than the outputSize
  * Optional Arguments :
   * `reduceStride` : strides of the 1x1 (reduction) convolutions. Defaults to `{1,1,...}`.
   * `transfer` : transfer function like `nn.Tanh`,`nn.Sigmoid`, `nn.ReLU`, `nn.Identity`, etc. It is used after each reduction (1x1 convolution) and convolution. Defaults to `nn.ReLU`.
   * `batchNorm` : set this to `true` to use batch normalization. Defaults to `false`. Note that batch normalization can be awesome
   * `padding` : set this to `true` to add padding to the input of the convolutions such that output width and height are same as that of the original non-padded `input`. Defaults to `true`.
   * `kernelSize` : size (`height = width`) of the non-1x1 convolution kernels. Defaults to `{5,3}`.
   * `kernelStride` : stride of the kernels (`height = width`) of the convolution. Defaults to `{1,1}`
   * `poolSize`: size (`height = width`) of the spatial max pooling used in the next-to-last column. Defaults to 3.
   * `poolStride` : stride (`height = width`) of the spatial max pooling. Defaults to 1.


For a complete example using this module, refer to the following :
 * [deep inception training script](https://github.com/nicholas-leonard/dp/blob/master/examples/deepinception.lua) ;
 * [openface facial recognition](https://github.com/cmusatyalab/openface) (the model definition is [here](https://github.com/cmusatyalab/openface/blob/master/models/openface/nn4.def.lua)).

<a name='nn.Collapse'></a>
## Collapse ##

```lua
module = nn.Collapse(nInputDim)
```

This module is the equivalent of:
```
view = nn.View(-1)
view:setNumInputDim(nInputDim)
```
It collapses all non-batch dimensions. This is useful for converting
a spatial feature map to the single dimension required by a dense
hidden layer like Linear.

<a name='nn.Convert'></a>
## Convert ##

```lua
module = nn.Convert([inputShape, outputShape])
```
Module to convert between different data formats.
For example, we can flatten images by using :
```lua
module = nn.Convert('bchw', 'bf')
```
or equivalently
```lua
module = nn.Convert('chw', 'f')
```
Lets try it with an input:
```lua
print(module:forward(torch.randn(3,2,3,1)))
 0.5692 -0.0190  0.5243  0.7530  0.4230  1.2483
-0.9142  0.6013  0.5608 -1.0417 -1.4014  1.0177
-1.5207 -0.1641 -0.4166  1.4810 -1.1725 -1.0037
[torch.DoubleTensor of size 3x6]
```
You could also try:

```lua
module = nn.Convert('chw', 'hwc')
input = torch.randn(1,2,3,2)
input:select(2,1):fill(1)
input:select(2,2):fill(2)
print(input)
(1,1,.,.) =
  1  1
  1  1
  1  1
(1,2,.,.) =
  2  2
  2  2
  2  2
[torch.DoubleTensor of size 1x2x3x2]
print(module:forward(input))
(1,1,.,.) =
  1  2
  1  2

(1,2,.,.) =
  1  2
  1  2

(1,3,.,.) =
  1  2
  1  2
[torch.DoubleTensor of size 1x3x2x2]
```


Furthermore, it automatically converts the `input` to have the same type as `self.output`
(i.e. the type of the module).
So you can also just use is for automatic input type converions:
```lua
module = nn.Convert()
print(module.output) -- type of module
[torch.DoubleTensor with no dimension]
input = torch.FloatTensor{1,2,3}
print(module:forward(input))
 1
 2
 3
[torch.DoubleTensor of size 3]
```

<a name='nn.ZipTable'></a>
## ZipTable ##

```lua
module = nn.ZipTable()
```

Zips a table of tables into a table of tables.

Example:
```lua
print(module:forward{ {'a1','a2'}, {'b1','b2'}, {'c1','c2'} })
{ {'a1','b1','c1'}, {'a2','b2','c2'} }
```

<a name='nn.ZipTableOneToMany'></a>
## ZipTableOneToMany ##

```lua
module = nn.ZipTableOneToMany()
```

Zips a table of element `el` and table of elements `tab` into a table of tables, where the i-th table contains the element `el` and the i-th element in table `tab`

Example:
```lua
print(module:forward{ 'el', {'a','b','c'} })
{ {'el','a'}, {'el','b'}, {'el','c'} }
```

<a name='nn.CAddTensorTable'></a>
## CAddTensorTable ##

```lua
module = nn.CAddTensorTable()
```

Adds the first element `el` of the input table `tab` to each tensor contained in the second element of `tab`, which is itself a table

Example:
```lua
print(module:forward{ (0,1,1), {(0,0,0),(1,1,1)} })
{ (0,1,1), (1,2,2) }
```


<a name='nn.ReverseTable'></a>
## ReverseTable ##

```lua
module = nn.ReverseTable()
```

Reverses the order of elements in a table.

Example:

```lua
print(module:forward{1,2,3,4})
{4,3,2,1}
```

<a name='nn.PrintSize'></a>
## PrintSize ##

```lua
module = nn.PrintSize(name)
```

This module is useful for debugging complicated module composites.
It prints the size of the `input` and `gradOutput` during `forward`
and `backward` propagation respectively.
The `name` is a string used to identify the module along side the printed size.

<a name='nn.Clip'></a>
## Clip ##

```lua
module = nn.Clip(minval, maxval)
```

This module clips `input` values such that the output is between `minval` and `maxval`.

<a name='nn.Constant'></a>
## Constant ##

```lua
module = nn.Constant(value, nInputDim)
```

This module outputs a constant value given an input.
If `nInputDim` is specified, it uses the input to determine the size of the batch.
The `value` is then replicated over the batch.
Otherwise, the `value` Tensor is output as is.
During `backward`, the returned `gradInput` is a zero Tensor of the same size as the `input`.
This module has no trainable parameters.

You can use this with nn.ConcatTable() to append constant inputs to an input :

```lua
nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity())
```

This is useful when you want to output a value that is independent of the
input to the neural network (see [this example](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)).

<a name='nn.SpatialUniformCrop'></a>
## SpatialUniformCrop ##

```lua
module = nn.SpatialUniformCrop(oheight, owidth)
```

During training, this module will output a cropped patch of size `oheight, owidth`
within the boundaries of the `input` image.
For each example, a location is sampled from a uniform distribution
such that each possible patch has an equal probability of being sampled.

During evaluation, the center patch is cropped and output.

This module is commonly used at the input layer to artificially
augment the size of the dataset to prevent overfitting.

<a name='nn.SpatialGlimpse'></a>
## SpatialGlimpse ##
Ref. A. [Recurrent Model for Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)

```lua
module = nn.SpatialGlimpse(size, depth, scale)
```

A glimpse is the concatenation of down-scaled cropped images of
increasing scale around a given location in a given image.
The input is a pair of Tensors: `{image, location}`
`location` are `(y,x)` coordinates of the center of the different scales
of patches to be cropped from image `image`.
Coordinates are between `(-1,-1)` (top-left) and `(1,1)` (bottom-right).
The `output` is a batch of glimpses taken in image at location `(y,x)`.

`size` can be either a scalar which specifies the `width = height` of glimpses,
or a table of `{height, width}` to support a rectangular shape of glimpses.
`depth` is number of patches to crop per glimpse (one patch per depth).
`scale` determines the `size(t) = scale * size(t-1)` of successive cropped patches.

So basically, this module can be used to focus the attention of the model
on a region of the input `image`.
It is commonly used with the [RecurrentAttention](https://github.com/Element-Research/rnn#rnn.RecurrentAttention)
module (see [this example](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)).

<a name='nn.WhiteNoise'></a>
## WhiteNoise ##

```lua
module = nn.WhiteNoise([mean, stdev])
```

Useful in training [Denoising Autoencoders] (http://arxiv.org/pdf/1507.02672v1.pdf).
Takes `mean` and `stdev` of the normal distribution as input.
Default values for mean and standard deviation are 0 and 0.1 respectively.
With `module:training()`, noise is added during forward.
During `backward` gradients are passed as it is.
With `module:evaluate()` the mean is added to the input.

<a name='nn.SpatialRegionDropout'></a>
## SpatialRegionDropout ##

```lua
module = nn.SpatialRegionDropout(p)
```
Following is an example of `SpatialRegionDropout` outputs on the famous lena image.

**Input**

![Lena](tutorials/lena.jpg)

**Outputs**

![Lena](tutorials/srd1.jpg)           ![Lena](tutorials/srd2.jpg)

<a name='nn.FireModule'></a>
## FireModule ##
Ref: http://arxiv.org/pdf/1602.07360v1.pdf
```lua
module = nn.FireModule(nInputPlane, s1x1, e1x1, e3x3, activation)
```
FireModule is comprised of two submodules 1) A *squeeze* convolution module comprised of `1x1` filters followed by 2) an *expand* module that is comprised of a mix of `1x1` and `3x3` convolution filters.
Arguments: `s1x1`: number of `1x1` filters in the squeeze submodule, `e1x1`: number of `1x1` filters in the expand submodule, `e3x3`: number of `3x3` filters in the expand submodule. It is recommended that `s1x1` be less than `(e1x1+e3x3)` if you want to limit the number of input channels to the `3x3` filters in the expand submodule.
FireModule works only with batches, for single sample convert the sample to a batch of size 1.

<a name='nn.SpatialFeatNormalization'></a>
## SpatialFeatNormalization ##
```lua
module = nn.SpatialFeatNormalization(mean, std)
```
This module normalizies each feature channel of input image based on its corresponding mean and standard deviation scalar values. This module does not learn the `mean` and `std`, they are provided as arguments.

<a name='nn.SpatialBinaryConvolution'></a>
## SpatialBinaryConvolution ##

```lua
module = nn.SpatialBinaryConvolution(nInputPlane, nOutputPlane, kW, kH)
```
Functioning of SpatialBinaryConvolution is similar to nn/SpatialConvolution. Only difference is that Binary weights are used for forward/backward and floating point weights are used for weight updates. Check **Binary-Weight-Network** section of [XNOR-net](http://arxiv.org/pdf/1603.05279v2.pdf).

<a name='nn.SimpleColorTransform'></a>
## SimpleColorTransform ##

```lua
range = torch.rand(inputChannels) -- Typically range is specified by user.
module = nn.SimpleColorTransform(inputChannels, range)
```
This module performs a simple data augmentation technique. SimpleColorTransform module adds random noise to each color channel independently. In more advanced data augmentation technique noise is added using principal components of color channels. For that please check **PCAColorTransform**

<a name='nn.PCAColorTransform'></a>
## PCAColorTransform ##

```lua
eigenVectors = torch.rand(inputChannels, inputChannels) -- Eigen Vectors
eigenValues = torch.rand(inputChannels) -- Eigen
std = 0.1 -- Std deviation of normal distribution with mean zero for noise.
module = nn.PCAColorTransform(inputChannels, eigenVectors, eigenValues, std)
```
This module performs a data augmentation using Principal Component analysis of pixel values. When in training mode, mulitples of principal components are added to input image pixels. Magnitude of value added (noise) is dependent upon the corresponding eigen value and a random value sampled from a Gaussian distribution with mean zero and `std` (default 0.1) standard deviation. This technique was used in the famous [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) paper.

<a name = 'nn.OneHot'></a>
## OneHot ##

```lua
module = nn.OneHot(outputSize)
```

Transforms a tensor of `input` indices having integer values between 1 and `outputSize` into
a tensor of one-hot vectors of size `outputSize`.

Forward an index to get a one-hot vector :

```lua
> module = nn.OneHot(5) -- 5 classes
> module:forward(torch.LongTensor{3})
 0  0  1  0  0
[torch.DoubleTensor of size 1x5]
```

Forward a batch of 3 indices. Notice that these need not be stored as `torch.LongTensor` :

```lua
> module:forward(torch.Tensor{3,2,1})
 0  0  1  0  0
 0  1  0  0  0
 1  0  0  0  0
[torch.DoubleTensor of size 3x5]
```

Forward batch of `2 x 3` indices :

```lua
oh:forward(torch.Tensor{{3,2,1},{1,2,3}})
(1,.,.) =
  0  0  1  0  0
  0  1  0  0  0
  1  0  0  0  0

(2,.,.) =
  1  0  0  0  0
  0  1  0  0  0
  0  0  1  0  0
[torch.DoubleTensor of size 2x3x5]
```

<a name='nn.Kmeans'></a>
## Kmeans ##

```lua
km = nn.Kmeans(k, dim)
```

`k` is the number of centroids and `dim` is the dimensionality of samples.
You can either initialize centroids randomly from input samples or by using *kmeans++* algorithm.

```lua
km:initRandom(samples) -- Randomly initialize centroids from input samples.
km:initKmeansPlus(samples) -- Use Kmeans++ to initialize centroids.
```

Example showing how to use Kmeans module to do standard Kmeans clustering.

```lua
attempts = 10
iter = 100 -- Number of iterations
bestKm = nil
bestLoss = math.huge
learningRate = 1
for j=1, attempts do
   local km = nn.Kmeans(k, dim)
   km:initKmeansPlus(samples)
   for i=1, iter do
      km:zeroGradParameters()
      km:forward(samples) -- sets km.loss
      km:backward(samples, gradOutput) -- gradOutput is ignored

      -- Gradient Descent weight/centroids update
      km:updateParameters(learningRate)
   end

   if km.loss < bestLoss then
      bestLoss = km.loss
      bestKm = km:clone()
   end
end
```
`nn.Kmeans()` module maintains loss only for the latest forward. If you want to maintain loss over the whole dataset then you who would need do it my adding the module loss for every forward.

You can also use `nn.Kmeans()` as an auxillary layer in your network.
A call to `forward` will generate an `output` containing the index of the nearest cluster for each sample in the batch.
The `gradInput` generated by `updateGradInput` will be zero.

<a name='nn.ModuleCriterion'></a>
## ModuleCriterion ##

```lua
criterion = nn.ModuleCriterion(criterion [, inputModule, targetModule, castTarget])
```

This criterion decorates a `criterion` by allowing the `input` and `target` to be
fed through an optional `inputModule` and `targetModule` before being passed to the
`criterion`. The `inputModule` must not contain parameters as these would not be updated.

When `castTarget = true` (the default), the `targetModule` is cast along with the `inputModule` and
`criterion`. Otherwise, the `targetModule` isn't.

<a name='nn.NCEModule'></a>
## NCEModule
Ref. A [RNNLM training with NCE for Speech Recognition](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)

```lua
ncem = nn.NCEModule(inputSize, outputSize, k, unigrams, [Z])
```

When used in conjunction with [NCECriterion](#nn.NCECriterion),
the `NCEModule` implements [noise-contrastive estimation](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf).

The point of the NCE is to speedup computation for large `Linear` + `SoftMax` layers.
Computing a forward/backward for `Linear(inputSize, outputSize)` for a large `outputSize` can be very expensive.
This is common when implementing language models having with large vocabularies of a million words.
In such cases, NCE can be an efficient alternative to computing the full `Linear` + `SoftMax` during training and
cross-validation.

The `inputSize` and `outputSize` are the same as for the `Linear` module.
The number of noise samples to be drawn per example is `k`. A value of 25 should work well.
Increasing it will yield better results, while a smaller value will be more efficient to process.
The `unigrams` is a tensor of size `outputSize` that contains the frequencies or probability distribution over classes.
It is used to sample noise samples via a fast implementation of `torch.multinomial`.
The `Z` is the normalization constant of the approximated SoftMax.
The default is `math.exp(9)` as specified in Ref. A.

For inference, or measuring perplexity, the full `Linear` + `SoftMax` will need to
be computed. The `NCEModule` can do this by switching on the following :

```lua
ncem:evaluate()
ncem.normalized = true
```

Furthermore, to simulate `Linear` + `LogSoftMax` instead, one need only add the following to the above:

```lua
ncem.logsoftmax = true
```

An example is provided via the rnn package.

<a name='nn.NCECriterion'></a>
## NCECriterion

```lua
ncec = nn.NCECriterion()
```

This criterion only works with an [NCEModule](#nn.NCEModule) on the output layer.
Together, they implement [noise-contrastive estimation](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf).


<a name='nn.Reinforce'></a>
## Reinforce ##
Ref A. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

Abstract class for modules that implement the REINFORCE algorithm (ref. A).

```lua
module = nn.Reinforce([stochastic])
```

The `reinforce(reward)` method is called by a special Reward Criterion (e.g. [VRClassReward](#nn.VRClassReward)).
After which, when backward is called, the reward will be used to generate gradInputs.
When `stochastic=true`, the module is stochastic (i.e. samples from a distribution)
during evaluation and training.
When `stochastic=false` (the default), the module is only stochastic during training.

The REINFORCE rule for a module can be summarized as follows :
```lua
            d ln(f(output,input))
gradInput = ---------------------  * reward
                  d input
```
where the `reward` is what is provided by a Reward criterion like
[VRClassReward](#nn.VRClassReward) via the [reinforce](#nn.Module.reinforce) method.
The criterion will normally be responsible for the following formula :
```lua
reward = a*(R - b)
```
where `a` is the alpha of the original paper, i.e. a reward scale,
`R` is the raw reward (usually 0 or 1), and `b` is the baseline reward,
which is often taken to be the expected raw reward `R`.

The `output` is usually sampled from a probability distribution `f()`
parameterized by the `input`.
See [ReinforceBernoulli](#nn.ReinforceBernoulli) for a concrete derivation.

Also, as you can see, the gradOutput is ignored. So within a backpropagation graph,
the `Reinforce` modules will replace the backpropagated gradients (`gradOutput`)
with their own obtained from the broadcasted `reward`.

<a name='nn.ReinforceBernoulli'></a>
## ReinforceBernoulli ##
Ref A. [Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

```lua
module = nn.ReinforceBernoulli([stochastic])
```

A [Reinforce](#nn.Reinforce) subclass that implements the REINFORCE algorithm
(ref. A p.230-236) for the Bernoulli probability distribution.
Inputs are bernoulli probabilities `p`.
During training, outputs are samples drawn from this distribution.
During evaluation, when `stochastic=false`, outputs are the same as the inputs.
Uses the REINFORCE algorithm (ref. A p.230-236) which is
implemented through the [reinforce](#nn.Module.reinforce) interface (`gradOutputs` are ignored).

Given the following variables :

 * `f` : bernoulli probability mass function
 * `x` : the sampled values (0 or 1) (i.e. `self.output`)
 * `p` : probability of sampling a 1

the derivative of the log bernoulli w.r.t. probability `p` is :
```
d ln(f(output,input))   d ln(f(x,p))    (x - p)
--------------------- = ------------ = ---------
      d input               d p         p(1 - p)
```

<a name='nn.ReinforceNormal'></a>
## ReinforceNormal ##
Ref A. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

```lua
module = nn.ReinforceNormal(stdev, [stochastic])
```

A [Reinforce](#nn.Reinforce) subclass that implements the REINFORCE algorithm
(ref. A p.238-239) for a Normal (i.e. Gaussian) probability distribution.
Inputs are the means of the normal distribution.
The `stdev` argument specifies the standard deviation of the distribution.
During training, outputs are samples drawn from this distribution.
During evaluation, when `stochastic=false`, outputs are the same as the inputs, i.e. the means.
Uses the REINFORCE algorithm (ref. A p.238-239) which is
implemented through the [reinforce](#nn.Module.reinforce) interface (`gradOutputs` are ignored).

Given the following variables :

  * `f` : normal probability density function
  * `x` : the sampled values (i.e. `self.output`)
  * `u` : mean (`input`)
  * `s` : standard deviation (`self.stdev`)

the derivative of log normal w.r.t. mean `u` is :
```
d ln(f(x,u,s))   (x - u)
-------------- = -------
     d u           s^2
```

As an example, it is used to sample locations for the [RecurrentAttention](https://github.com/Element-Research/rnn#rnn.RecurrentAttention)
module (see [this example](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)).

<a name='nn.ReinforceGamma'></a>
## ReinforceGamma ##
Ref A. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

```lua
module = nn.ReinforceGamma(scale, [stochastic])
```

A [Reinforce](#nn.Reinforce) subclass that implements the REINFORCE algorithm
(ref. A) for a [Gamma probability distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
parametrized by shape (k) and scale (theta) variables.
Inputs are the shapes of the gamma distribution.
During training, outputs are samples drawn from this distribution.
During evaluation, when `stochastic=false`, outputs are equal to the mean, defined as the product of
shape and scale ie. `k*theta`.
Uses the REINFORCE algorithm (ref. A) which is
implemented through the [reinforce](#nn.Module.reinforce) interface (`gradOutputs` are ignored).

Given the following variables :

  * `f` : gamma probability density function
  * `g` : digamma function
  * `x` : the sampled values (i.e. `self.output`)
  * `k` : shape (`input`)
  * `t` : scale

the derivative of log gamma w.r.t. shape `k` is :
```
d ln(f(x,k,t))
-------------- = ln(x) - g(k) - ln(t)
      d k
```

<a name='nn.ReinforceCategorical'></a>
## ReinforceCategorical ##
Ref A. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

```lua
module = nn.ReinforceCategorical([stochastic])
```

A [Reinforce](#nn.Reinforce) subclass that implements the REINFORCE algorithm
(ref. A) for a Categorical (i.e. Multinomial with one sample) probability distribution.
Inputs are the categorical probabilities of the distribution : `p[1], p[2], ..., p[k]`.
These are usually the output of a SoftMax.
For `n` categories, both the `input` and `output` ares of size `batchSize x n`.
During training, outputs are samples drawn from this distribution.
The outputs are returned in one-hot encoding i.e.
the output for each example has exactly one category having a 1, while the remainder are zero.
During evaluation, when `stochastic=false`, outputs are the same as the inputs, i.e. the probabilities `p`.
Uses the REINFORCE algorithm (ref. A) which is
implemented through the [reinforce](#nn.Module.reinforce) interface (`gradOutputs` are ignored).


Given the following variables :

  * `f` : categorical probability mass function
  * `x` : the sampled indices (one per sample) (`self.output` is the one-hot encoding of these indices)
  * `p` : probability vector (`p[1], p[2], ..., p[k]`) (`input`)

the derivative of log categorical w.r.t. probability vector `p` is :
```
d ln(f(x,p))     1/p[i]    if i = x
------------ =
    d p          0         otherwise
```

<a name='nn.VRClassReward'></a>
## VRClassReward ##
Ref A. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)

This Reward criterion implements the REINFORCE algoritm (ref. A) for classification models.
Specifically, it is a Variance Reduces (VR) classification reinforcement leanring (reward-based) criterion.

```lua
vcr = nn.VRClassReward(module [, scale, criterion])
```

While it conforms to the Criterion interface (which it inherits),
it does not backpropagate gradients (except for the baseline `b`; see below).
Instead, a `reward` is broadcast to the `module` via the [reinforce](#nn.Module.reinforce) method.

The criterion implements the following formula :
```lua
reward = a*(R - b)
```
where `a` is the alpha described in Ref. A, i.e. a reward `scale` (defaults to 1),
`R` is the raw reward (0 for incorrect and 1 for correct classification),
and `b` is the baseline reward, which is often taken to be the expected raw reward `R`.

The `target` of the criterion is a tensor of class indices.
The `input` to the criterion is a table `{y,b}` where `y` is the probability
(or log-probability) of classes (usually the output of a SoftMax),
and `b` is the baseline reward discussed above.

For each example, if `argmax(y)` is equal to the `target` class, the raw reward `R = 1`, otherwize `R = 0`.

As for `b`, its `gradInputs` are obtained from the `criterion`, which defaults to `MSECriterion`.
The `criterion`'s target is the commensurate raw reward `R`.
Using `a*(R-b)` instead of `a*R` to obtain a `reward` is what makes this class variance reduced (VR).
By reducing the variance, the training can converge faster (Ref. A).
The predicted `b` can be nothing more than the expectation `E(R)`.

Note : for RNNs with R = 1 for last step in sequence, encapsulate it
in `nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))`.

For an example, this criterion is used along with the [RecurrentAttention](https://github.com/Element-Research/rnn#rnn.RecurrentAttention)
module to [train a recurrent model for visual attention](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua).

<a name='nn.BinaryClassReward'></a>
## BinaryClassReward ##

```lua
bcr = nn.BinaryClassReward(module [, scale, criterion])
```

This module implements [VRClassReward](#nn.VRClassReward) for binary classification problems.
So basically, the `input` is still a table of two tensors.
The first input tensor is of size `batchsize` containing Bernoulli probabilities.
The second input tensor is the baseline prediction described in `VRClassReward`.
The targets contain zeros and ones.

<a name='nn.BLR'></a>
## BinaryLogisticRegression ##
Ref A. [Learning to Segment Object Candidates](http://arxiv.org/pdf/1506.06204v2.pdf)
This criterion implements the score criterion mentioned in (ref. A).

```lua
criterion = nn.BinaryLogisticRegression()
```

BinaryLogisticRegression implements following cost function for binary classification.

```

 log( 1 + exp( -y_k * score(x_k) ) )

```
where `y_k` is binary target `score(x_k)` is the corresponding prediction. `y_k` has value `{-1, +1}` and `score(x_k)` has value in `[-1, +1]`.

<a name='nn.SpatialBLR'></a>
## SpatialBinaryLogisticRegression ##
Ref A. [Learning to Segment Object Candidates](http://arxiv.org/pdf/1506.06204v2.pdf)

This criterion implements the spatial component of the criterion mentioned in  (ref. A).

```lua
criterion = nn.SpatialBinaryLogisticRegression()
```

SpatialBinaryLogisticRegression implements following cost function for binary pixel classification.
```
   1
_______ sum_ij [ log( 1 + exp( -m_ij * f_ij ) ) ]
 2*w*h
```
where `m_ij` is target binary image and `f_ij` is the corresponding prediction. `m_ij` has value `{-1, +1}` and `f_ij` has value in `[-1, +1]`.

