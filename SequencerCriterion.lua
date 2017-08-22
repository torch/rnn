------------------------------------------------------------------------
--[[ SequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
-- WARNING : assumes that the decorated criterion is stateless, i.e.
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local SequencerCriterion, parent = torch.class('nn.SequencerCriterion', 'nn.AbstractSequencerCriterion')

function SequencerCriterion:updateOutput(input, target)
   self.output = 0
   local seqlen
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      seqlen = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting target table")
      assert(#target == #input, "target should have as many elements as input")
      seqlen = #input
   end


   for i=1,seqlen do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input[i], target[i])
   end

   if self.sizeAverage then
      self.output = self.output / seqlen
   end

   return self.output
end

function SequencerCriterion:updateGradInput(input, target)
   local seqlen
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      seqlen = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting gradOutput table")
      assert(#target == #input, "target should have as many elements as input")
      seqlen = #input
   end

   local tableGradInput = {}
   for i=1,seqlen do
      local criterion = self:getStepCriterion(i)
      tableGradInput[i] = criterion:backward(input[i], target[i])
   end

   if self.sizeAverage then
      nn.utils.recursiveDiv(tableGradInput, seqlen)
   end

   if torch.isTensor(input) then
      self.gradInput = tableGradInput[1].new()
      self.gradInput:resize(seqlen, unpack(tableGradInput[1]:size():totable()))
      for step=1,seqlen do
         self.gradInput[step]:copy(tableGradInput[step])
      end
   else
      self.gradInput = tableGradInput
   end

   return self.gradInput
end
