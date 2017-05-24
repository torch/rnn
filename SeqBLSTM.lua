------------------------------------------------------------------------
--[[ SeqBLSTM ]] --
-- Bi-directional LSTM using two SeqLSTM modules.
-- Input is a tensor size: seqlen x batchsize x inputsize.
-- Output is a tensor size: seqlen x batchsize x outputsize.
-- Applies a forward LSTM to input tensor in forward order
-- and applies a backward LSTM in reverse order.
-- Reversal of the sequence happens on the time dimension.
-- For each step, the outputs of both LSTMs are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
------------------------------------------------------------------------
local SeqBLSTM, parent = torch.class('nn.SeqBLSTM', 'nn.BiSequencer')

function SeqBLSTM:__init(inputsize, hiddensize, outputsize, merge)
   if torch.isTypeOf(outputsize, 'nn.Module') then
      merge = outputsize
      outputsize = nil
   end
   parent.__init(self, nn.SeqLSTM(inputsize, hiddensize, outputsize), nil, merge)
end
