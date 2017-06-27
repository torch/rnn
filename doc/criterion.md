# Criterion Modules #
<a name='rnn.criterion'></a>

Criterions used for handling sequential inputs and targets :
 * [AbstractSequencerCriterion](criterion.md#rnn.AbstractSequencerCriterion) : abstact class for criterions that handle sequences (tensor or table);
 * [SequencerCriterion](criterion.md#rnn.SequencerCriterion) : sequentially applies the same criterion to a sequence of inputs and targets;
 * [RepeaterCriterion](criterion.md#rnn.RepeaterCriterion) : repeatedly applies the same criterion with the same target on a sequence.


<a name='rnn.AbstractSequencerCriterion'></a>
## AbstractSequencerCriterion ##

```lua
asc = nn.AbstractSequencerCriterion(stepcriterion, [sizeAverage])
```

Similar to the `stepmodule` passed to the [AbstractRecurrent](recurrent.md#rnn.AbstractRecurrent) constructor,
the `stepcriterion` is internally cloned for each time-step.
Unlike the `stepmodule` the `stepcriterion` never has any parameters to share.

<a name='rnn.AbstractSequencerCriterion.getStepCriterion'></a>
### [criterion] getStepCriterion(step)

Returns a `criterion` clone of the `stepcriterion` (stored in `self.clones[1]`) for a specific time-`step`.

<a name='rnn.AbstractSequencerCriterion.setZeroMask'></a>
### setZeroMask(zeroMask)

Expects a `seqlen x batchsize` `zeroMask`.
The `zeroMask` is then passed to `seqlen` criterions by indexing `zeroMask[step]`.
When `zeroMask=false`, the zero-masking is disabled.

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
This is useful when used in conjunction with a [Sequencer](sequencer.md#rnn.Sequencer).

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
