require 'torchx'
local _ = require 'moses'
require 'nn'

-- create global rnn table:
rnn = {}
rnn.version = 2.4 -- merge dpnn

-- lua 5.2 compat

function nn.require(packagename)
   assert(torch.type(packagename) == 'string')
   local success, message = pcall(function() require(packagename) end)
   if not success then
      print("missing package "..packagename..": run 'luarocks install nnx'")
      error(message)
   end
end


-- c lib:
require "paths"
paths.require 'librnn'

unpack = unpack or table.unpack

require('rnn.recursiveUtils')
require('rnn.utils')

-- extensions to existing nn.Module
require('rnn.Module')
require('rnn.Container')
require('rnn.Sequential')
require('rnn.ParallelTable')
require('rnn.LookupTable')
require('rnn.Dropout')

-- extensions to existing criterions
require('rnn.Criterion')

-- decorator modules
require('rnn.Decorator')
require('rnn.Serial')
require('rnn.DontCast')
require('rnn.NaN')
require('rnn.Profile')

-- extensions to make serialization more efficient
require('rnn.SpatialMaxPooling')
require('rnn.SpatialConvolution')
require('rnn.SpatialConvolutionMM')
require('rnn.SpatialBatchNormalization')
require('rnn.BatchNormalization')


-- modules
require('rnn.PrintSize')
require('rnn.Convert')
require('rnn.Constant')
require('rnn.Collapse')
require('rnn.ZipTable')
require('rnn.ZipTableOneToMany')
require('rnn.CAddTensorTable')
require('rnn.ReverseTable')
require('rnn.Dictionary')
require('rnn.Inception')
require('rnn.Clip')
require('rnn.SpatialUniformCrop')
require('rnn.SpatialGlimpse')
require('rnn.WhiteNoise')
require('rnn.ArgMax')
require('rnn.CategoricalEntropy')
require('rnn.TotalDropout')
require('rnn.Kmeans')
require('rnn.OneHot')
require('rnn.SpatialRegionDropout')
require('rnn.FireModule')
require('rnn.SpatialFeatNormalization')
require('rnn.ZeroGrad')
require('rnn.LinearNoBias')
require('rnn.SAdd')
require('rnn.CopyGrad')
require('rnn.VariableLength')
require('rnn.StepLSTM')
require('rnn.LookupTableMaskZero')
require('rnn.MaskZero')
require('rnn.TrimZero')
require('rnn.SpatialBinaryConvolution')
require('rnn.SimpleColorTransform')
require('rnn.PCAColorTransform')

-- Noise Contrastive Estimation
require('rnn.NCEModule')
require('rnn.NCECriterion')

-- REINFORCE
require('rnn.Reinforce')
require('rnn.ReinforceGamma')
require('rnn.ReinforceBernoulli')
require('rnn.ReinforceNormal')
require('rnn.ReinforceCategorical')

-- REINFORCE criterions
require('rnn.VRClassReward')
require('rnn.BinaryClassReward')

-- criterions
require('rnn.ModuleCriterion')
require('rnn.BinaryLogisticRegression')
require('rnn.SpatialBinaryLogisticRegression')

-- for testing:
require('rnn.test')
require('rnn.bigtest')

-- recurrent modules
require('rnn.AbstractRecurrent')
require('rnn.Recursor')
require('rnn.Recurrence')
require('rnn.LinearRNN')
require('rnn.LookupRNN')
require('rnn.LSTM')
require('rnn.RecLSTM')
require('rnn.GRU')
require('rnn.Mufuru')
require('rnn.NormStabilizer')

-- sequencer modules
require('rnn.AbstractSequencer')
require('rnn.Repeater')
require('rnn.Sequencer')
require('rnn.BiSequencer')
require('rnn.BiSequencerLM')
require('rnn.RecurrentAttention')

-- sequencer + recurrent modules
require('rnn.SeqLSTM')
require('rnn.SeqLSTMP')
require('rnn.SeqGRU')
require('rnn.SeqReverseSequence')
require('rnn.SeqBRNN')

-- recurrent criterions:
require('rnn.SequencerCriterion')
require('rnn.RepeaterCriterion')
require('rnn.MaskZeroCriterion')

-- deprecated modules
require('rnn.FastLSTM')
require('rnn.Recurrent')

-- prevent likely name conflicts
nn.rnn = rnn

return rnn