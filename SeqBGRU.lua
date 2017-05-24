------------------------------------------------------------------------
--[[ SeqBGRU ]] --
-- Bi-directional GRU using two SeqGRU modules.
-- Input is a tensor size: seqlen x batchsize x inputsize.
-- Output is a tensor size: seqlen x batchsize x outputsize.
-- Applies a forward GRU to input tensor in forward order
-- and applies a backward GRU in reverse order.
-- Reversal of the sequence happens on the time dimension.
-- For each step, the outputs of both GRUs are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
------------------------------------------------------------------------
local SeqBGRU, parent = torch.class('nn.SeqBGRU', 'nn.BiSequencer')

function SeqBGRU:__init(inputsize, outputsize, merge)
   parent.__init(self, nn.SeqGRU(inputsize, outputsize), nil, merge)
end
