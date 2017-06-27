# REINFORCE modules #

<a name='rnn.reinforce'></a>
The following modules and criterions can be used to implement the REINFORCE algorithm :

 * [Reinforce](reinforce.md#nn.Reinforce) : abstract class for REINFORCE modules;
 * [ReinforceBernoulli](reinforce.md#nn.ReinforceBernoulli) : samples from Bernoulli distribution;
 * [ReinforceNormal](reinforce.md#nn.ReinforceNormal) : samples from Normal distribution;
 * [ReinforceGamma](reinforce.md#nn.ReinforceGamma) : samples from Gamma distribution;
 * [ReinforceCategorical](reinforce.md#nn.ReinforceCategorical) : samples from Categorical (Multinomial with one sample) distribution;
 * [VRClassReward](reinforce.md#nn.VRClassReward) : criterion for variance-reduced classification-based reward;
 * [BinaryClassReward](reinforce.md#nn.BinaryClassReward) : criterion for variance-reduced binary classification reward (like `VRClassReward`, but for binary classes);


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

<a name='nn.ReverseSequence'></a>
## ReverseSequence ##

```lua
module = nn.ReverseSequence()
```

Reverses the order of elements in a sequence table or a tensor.

Example using table:

```lua
print(module:forward{1,2,3,4})
{4,3,2,1}
```

Example using tensor:

```lua
print(module:forward(torch.Tensor({1,2,3,4})))
 4
 3
 2
 1
[torch.DoubleTensor of size 4]

```

<a name='nn.ReverseUnreverse'></a>
## ReverseUnreverse ##

```lua
ru = nn.ReverseUnreverse(sequencer)
```

This module is used internally by the [BiSequencer](sequencer.md#rnn.BiSequencer) module.
The `ReverseUnreverse` decorates a `sequencer` module like [SeqLSTM](sequencer.md#rnn.SeqLSTM) and [Sequencer](sequencer.md#rnn.Sequencer)
The `sequencer` module is expected to implement the [AbstractSequencer](sequencer.md#rnn.AbstractSequencer) interface.
When calling `forward`, the `seqlen x batchsize [x ...]` `input` tensor is reversed using [ReverseSequence](sequencer.md#rnn.ReverseSequence).
Then the `input` sequences are forwarded (in reverse order) through the `sequencer`.
The resulting `sequencer.output` sequences are reversed with respect to the `input`.
Before being returned to the caller, these are unreversed using another `ReverseSequence`.

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
