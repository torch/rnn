# Torch recurrent neural networks #

This is a recurrent neural network (RNN) library that extends Torch's nn.
You can use it to build RNNs, LSTMs, GRUs, BRNNs, BLSTMs, and so forth and so on.

This library includes documentation for the following objects :

  * [Recurrent modules](doc/recurrent.md) consider successive calls to `forward` as different time-steps in a sequence.
  * [Sequencer modules](doc/sequencer.md) `forward` entire sequences through a decorated `AbstractRecurrent` instance.
  * [Sequencer criterion](doc/criterion.md) handle `input` and `target` sequences.
  * [Miscellaneous modules and criterions](doc/miscellaneous.md) for things like noise contrastive estimation (NCE) and zero-masking.
  * [Reinforce modules and criterions](doc/reinforce.md) implement the REINFORCE learning rule.
	
	
<a name='rnn.examples'></a>
## Examples ##

A complete list of examples is available in the [examples directory](examples/README.md)

## Citation ##

If you use __rnn__ in your work, we'd really appreciate it if you could cite the following paper:

Léonard, Nicholas, Sagar Waghmare, Yang Wang, and Jin-Hwa Kim. [rnn: Recurrent Library for Torch.](http://arxiv.org/abs/1511.07889) arXiv preprint arXiv:1511.07889 (2015).

Any significant contributor to the library will also get added as an author to the paper.
A [significant contributor](https://github.com/torch/rnn/graphs/contributors)
is anyone who added at least 300 lines of code to the library.

## Troubleshooting ##

Most issues can be resolved by updating the various dependencies:
```bash
luarocks install torch
luarocks install nn
luarocks install torchx
luarocks install dataload
```

If you are using CUDA :
```bash
luarocks install cutorch
luarocks install cunn
```

And don't forget to update this package :
```bash
luarocks install rnn
```

If that doesn't fix it, open an issue on github.
