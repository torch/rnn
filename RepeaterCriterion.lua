------------------------------------------------------------------------
--[[ RepeaterCriterion ]]--
-- Applies a criterion to each of the inputs in a Table using the
-- same target (the target is repeated).
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
local RepeaterCriterion, parent = torch.class('nn.RepeaterCriterion', 'nn.AbstractSequencerCriterion')

function RepeaterCriterion:updateOutput(input, target)
   self.output = 0
   local seqlen
   if torch.isTensor(input) then
      seqlen = input:size(1)
   else
      seqlen = #input
   end

   for i=1,seqlen do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input[i], target)
   end


   if self.sizeAverage then
      self.output = self.output / seqlen
   end

   return self.output
end

function RepeaterCriterion:updateGradInput(input, target)
   self.gradInput = {}
   if torch.isTensor(input) then
      seqlen = input:size(1)
   else
      seqlen = #input
   end

   local tableGradInput = {}
   for i=1,seqlen do
      local criterion = self:getStepCriterion(i)
      tableGradInput[i] = criterion:backward(input[i], target)
   end

   if self.sizeAverage then
      nn.utils.recursiveDiv(tableGradInput[i], seqlen)
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
