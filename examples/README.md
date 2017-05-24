# Examples

This document outlines the variety of training scripts and external resources.

## Advanced training scripts

This section lists advanced training scripts that train RNNs on real-world datasets.
 1. [recurrent-language-model.lua](recurrent-language-model.lua): trains a stack of LSTM, GRU, MuFuRu, or Simple RNN on the Penn Tree Bank dataset without or without dropout.
 2. [recurrent-visual-attention.lua](recurrent-visual-attention.lua): training script used in [Recurrent Model for Visual Attention](http://torch.ch/blog/2015/09/21/rmva.html). Implements the REINFORCE learning rule to learn an attention mechanism for classifying MNIST digits, sometimes translated. Showcases `nn.RecurrentAttention`, `nn.SpatialGlimpse` and `nn.Reinforce`.
 3. [noise-contrastive-esimate.lua](noise-contrastive-estimate.lua): one of two training scripts used in [Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html). Single-GPU script for training recurrent language models on the Google billion words dataset. This example showcases version 2 zero-masking. Version 2 is more efficient than version 1 as the `zeroMask` is interpolated only once.
 4. [multigpu-nce-rnnlm.lua](multigpu-nce-rnnlm.lua) : 4-GPU version of `noise-contrastive-estimate.lua` for training larger multi-GPU models. Two of two training scripts used in the [Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html). This script is for training multi-layer [SeqLSTM](/README.md#rnn.SeqLSTM) language models on the [Google Billion Words dataset](https://github.com/Element-Research/dataload#dl.loadGBW). The example uses [MaskZero](/README.md#rnn.MaskZero) to train independent variable length sequences using the [NCEModule](/README.md#nn.NCEModule) and [NCECriterion](/README.md#nn.NCECriterion). This script is our fastest yet boasting speeds of 20,000 words/second (on NVIDIA Titan X) with a 2-layer LSTM having 250 hidden units, a batchsize of 128 and sequence length of a 100. Note that you will need to have [Torch installed with Lua instead of LuaJIT](http://torch.ch/docs/getting-started.html#_);
 5. [twitter-sentiment-rnn.lua](twitter-sentiment-rnn.lua) : trains stack of RNNs on a twitter sentiment analysis. The problem is a text classification problem that uses a sequence-to-one architecture. In this architecture, only the last RNN's last time-step is used for classification.

## Simple training scripts

This section lists simple training scripts that train RNNs on dummy datasets.
These scripts showcases the fundamental principles of the package.
 1. [simple-recurrent-network.lua](simple-recurrent-network.lua): uses the `nn.LookupRNN` module to instantiate a Simple RNN. Illustrates the first AbstractRecurrent instance in action. It has since been surpassed by the more flexible `nn.Recursor` and `nn.Recurrence`. The `nn.Recursor` class decorates any module to make it conform to the nn.AbstractRecurrent interface. The `nn.Recurrence` implements the recursive `h[t] <- forward(h[t-1], x[t])`. Together, `nn.Recursor` and `nn.Recurrence` can be used to implement a wide range of experimental recurrent architectures.
 2. [simple-sequencer-network.lua](simple-sequencer-network.lua): uses the `nn.Sequencer` module to accept a batch of sequences as `input` of size `seqlen x batchsize x ...`. Both tables and tensors are accepted as input and produce the same type of output (table->table, tensor->tensor). The `Sequencer` class abstract away the implementation of back-propagation through time. It also provides a `remember(['neither','both'])` method for triggering what the `Sequencer` remembers between iterations (forward,backward,update).
 3. [simple-recurrence-network.lua](simple-recurrence-network.lua): uses the `nn.Recurrence` module to define the h[t] <- sigmoid(h[t-1], x[t]) Simple RNN. Decorates it using `nn.Sequencer` so that an entire batch of sequences (`input`) can forward and backward propagated per update.
 4. [simple-bisequencer-network.lua](simple-bisequencer-network.lua): uses a `nn.BiSequencerLM` and two `nn.LookupRNN` to implement a simple bi-directional language model.
 5. [simple-bisequencer-network-variable.lua](simple-bisequencer-network-variable.lua): uses `nn.RecLSTM`, `nn.LookupTableMaskZero`, `nn.ZipTable`, `nn.MaskZero` and `nn.MaskZeroCriterion` to implement a simple bi-directional LSTM language model. This example uses version 1 zero-masking where the `zeroMask` is automatically interpolated from the `input`.
 6. [sequence-to-one.lua](sequence-to-one.lua): a simple sequence-to-one example that uses `Recurrence` to build an RNN and `SelectTable(-1)` to select the last time-step for discriminating the sequence.
 7. [encoder-decoder-coupling.lua](encoder-decoder-coupling.lua): uses two stacks of `nn.SeqLSTM` to implement an encoder and decoder. The final hidden state of the encoder initializes the hidden state of the decoder. Example of sequence-to-sequence learning.
 8. [nested-recurrence-lstm.lua](nested-recurrence-lstm.lua): demonstrates how RNNs can be nested to form complex RNNs.
 9. [recurrent-time-series.lua](recurrent-time-series.lua) demonstrates how train a simple RNN to do multi-variate time-series predication.

 ## External resources

  * [rnn-benchmarks](https://github.com/glample/rnn-benchmarks) : benchmarks comparing Torch (using this library), Theano and TensorFlow.
  * [dataload](https://github.com/Element-Research/dataload) : a collection of torch dataset loaders;
  * A brief (1 hours) overview of Torch7, which includes some details about the __rnn__ packages (at the end), is available via this [NVIDIA GTC Webinar video](http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4). In any case, this presentation gives a nice overview of Logistic Regression, Multi-Layer Perceptrons, Convolutional Neural Networks and Recurrent Neural Networks using Torch7;
