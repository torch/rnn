<a name='rnn.miscellaneous'></a>
# Miscellaneous Modules #


Miscellaneous modules and criterions :
 * [MaskZero](miscellaneous.md#rnn.MaskZero) : zeroes the `output` and `gradOutput` rows of the decorated module for commensurate
   * `input` rows which are tensors of zeros (version 1);
   * `zeroMask` elements which are 1 (version 2);
 * [LookupTableMaskZero](miscellaneous.md#rnn.LookupTableMaskZero) : extends `nn.LookupTable` to support zero indexes for padding. Zero indexes are forwarded as tensors of zeros;
 * [MaskZeroCriterion](miscellaneous.md#rnn.MaskZeroCriterion) : zeros the `gradInput` and `loss` rows of the decorated criterion for commensurate
   * `input` rows which are tensors of zeros (version 1);
   * `zeroMask` elements which are 1 (version 2);
 * [ReverseSequence](miscellaneous.md#nn.ReverseSequence) : reverse the order of elements in a sequence (table or tensor);
 * [ReverseUnreverse](miscellaneous.md#nn.ReverseUnreverse) : used internally by `nn.BiSequencer` for decorating `bwd` RNN.
 * [SpatialGlimpse](miscellaneous.md#nn.SpatialGlimpse) : takes a fovead glimpse of an image at a given location;
 * [NCEModule](miscellaneous.md#nn.NCEModule) : optimized placeholder for a `Linear` + `SoftMax` using [noise-contrastive estimation](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf).
 * [NCECriterion](miscellaneous.md#nn.NCECriterion) : criterion exclusively used with [NCEModule](#nn.NCEModule);
 * [VariableLength](miscellaneous.md#rnn.VariableLength): decorates a `Sequencer` to accept and produce a table of variable length inputs and outputs;
 * [dpnn modules](nn.Module): The Module interface has been further extended with methods that facilitate stochastic gradient descent like [updateGradParameters](#nn.Module.updateGradParameters) (for momentum learning), [weightDecay](#nn.Module.weightDecay), [maxParamNorm](#nn.Module.maxParamNorm) (for regularization), and so on.



<a name='rnn.MaskZero'></a>
## MaskZero ##

This module implements *zero-masking*.
Zero-masking implements the zeroing specific rows/samples of a module's `output` and `gradInput` states.
Zero-masking is used for efficiently processing variable length sequences.

```lua
mz = nn.MaskZero(module, [v1, maskinput, maskoutput])
```

This module zeroes the `output` and `gradOutput` rows of the decorated `module` where
 * the commensurate row of the `input` is a tensor of zeros (version 1 with `v1=true`); or
 * the commensurate element of the `zeroMask` tensor is 1 (version 2 with `v1=false`, the default).

Version 2 (the default), requires that [`setZeroMask(zeroMask)`](#rnn.MaskZero.setZeroMask)
be called beforehand. The `zeroMask` must be a `torch.ByteTensor` or `torch.CudaByteTensor` of size `batchsize`.

At a given time-step `t`, a sample `i` is masked when:
 * the `input[i]` is a row of zeros (version 1) where `input` is a batched time-step; or
 * the `zeroMask[{t,i}] = 1` (version 2).

When a sample time-step is masked, the hidden state is effectively reset (that is, forgotten) for the next non-mask time-step.
In other words, it is possible seperate unrelated sequences with a masked element.

When `maskoutput=true` (the default), `output` and `gradOutput` are zero-masked.
When `maskinput=true` (not the default), `input` and `gradInput` aere zero-masked.

Zero-masking only supports batch mode.

Caveat: `MaskZero` does not guarantee that the `output` and `gradOutput` tensors of the internal modules
of the decorated `module` will be zeroed.
`MaskZero` only affects the immediate `gradOutput` and `output` of the module that it encapsulates.
However, for most modules, the gradient update for that time-step will be zero because
backpropagating a gradient of zeros will typically yield zeros all the way to the input.
In this respect, modules that shouldn't be encapsulated inside a `MaskZero` are `AbsractRecurrent`
instances as the flow of gradients between different time-steps internally.
Instead, call the [AbstractRecurrent.maskZero](recurrent.md#rnn.AbstractRecurrent.maskZero) method
to encapsulate the internal `stepmodule`.

See the [noise-contrastive-estimate.lua](examples/noise-contrastive-estimate.lua) script for an example implementation of version 2 zero-masking.
See the [simple-bisequencer-network-variable.lua](examples/simple-bisequencer-network-variable.lua) script for an example implementation of version 1 zero-masking.

<a name='rnn.MaskZero.setZeroMask'></a>
### setZeroMask(zeroMask) ##

Set the `zeroMask` of the `MaskZero` module (required for version 2 forwards).
For example,
```lua
batchsize = 3
inputsize, outputsize = 2, 1
-- an nn.Linear module decorated with MaskZero (version 2)
module = nn.MaskZero(nn.Linear(inputsize, outputsize))
-- zero-mask the second sample/row
zeroMask = torch.ByteTensor(batchsize):zero()
zeroMask[2] = 1
module:setZeroMask(zeroMask)
-- forward
input = torch.randn(batchsize, inputsize)
output = module:forward(input)
print(output)
 0.6597
 0.0000
 0.8170
[torch.DoubleTensor of size 3x1]
```
The `output` is indeed zeroed for the second sample (`zeroMask[2] = 1`).
The `gradInput` would also be zeroed in the same way because the `gradOutput` would be zeroed:
```lua
gradOutput = torch.randn(batchsize, outputsize)
gradInput = module:backward(input, gradOutput)
print(gradInput)
 0.8187  0.0534
 0.0000  0.0000
 0.1742  0.0114
[torch.DoubleTensor of size 3x2]
```

For `Container` modules, a call to `setZeroMask()` is propagated to all component modules that expect a `zeroMask`.

When `zeroMask=false`, the zero-masking is disabled.

<a name='rnn.LookupTableMaskZero'></a>
## LookupTableMaskZero ##
This module extends `nn.LookupTable` to support zero indexes. Zero indexes are forwarded as zero tensors.

```lua
lt = nn.LookupTableMaskZero(nIndex, nOutput)
```

The `output` Tensor will have each row zeroed when the commensurate row of the `input` is a zero index.

This lookup table makes it possible to pad sequences with different lengths in the same batch with zero vectors.

Note that this module ignores version 2 zero-masking, and therefore expects inputs to be zeros where needed.

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


<a name='nn.Module'></a>
## Module ##

The Module interface has been further extended with methods that facilitate
stochastic gradient descent like [updateGradParameters](#nn.Module.updateGradParameters) (for momentum learning),
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
This is particularly useful for [Recurrent neural networks](https://github.com/torch/rnn/blob/master/doc/README.md)
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
