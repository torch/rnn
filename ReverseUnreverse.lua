----------------------------------------------------------
--[[ ReverseUnreverse ]]--
-- This module is used internally by BiSequencer modules
-- to handle the backward sequence.
-- It reverses the input and output sequences and
-- reverses the zeroMask.
----------------------------------------------------------
local ReverseUnreverse, parent = torch.class("nn.ReverseUnreverse", "nn.Decorator")

function ReverseUnreverse:__init(sequencer)
   assert(nn.BiSequencer.isSeq(sequencer), "Expecting AbstractSequencer or SeqLSTM or SeqGRU at arg 1")
   parent.__init(self, nn.Sequential()
      :add(nn.ReverseSequence()) -- reverse
      :add(sequencer)
      :add(nn.ReverseSequence()) -- unreverse
   )
end

function ReverseUnreverse:setZeroMask(zeroMask)
   -- reverse the zeroMask
   assert(torch.isTensor(zeroMask))
   assert(zeroMask:dim() >= 2)
   self._zeroMask = self._zeroMask or zeroMask.new()
   self._zeroMask:resizeAs(zeroMask)
   self._range = self._range or torch.isCudaTensor(zeroMask) and torch.CudaLongTensor() or torch.LongTensor()
   local seqlen = zeroMask:size(1)
   if self._range:nElement() ~= seqlen then
      self._range:range(seqlen, 1, -1)
   end

   self._zeroMask:index(zeroMask, 1, self._range)
   self.modules[1]:setZeroMask(self._zeroMask)
end

function ReverseUnreverse:reinforce(zeroMask)
   error"Not implemented"
end

function ReverseUnreverse:clearState()
   self._zeroMask = nil
   return parent.clearState(self)
end

function ReverseUnreverse:type(...)
   self:clearState()
   return parent.type(self, ...)
end

function ReverseUnreverse:getModule()
   return self:get(1):get(2)
end