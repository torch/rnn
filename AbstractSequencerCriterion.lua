------------------------------------------------------------------------
--[[ AbstractSequencerCriterion ]]--
-- Inherited by SequencerCriterion and RepeaterCriterion
-- WARNING : assumes that the decorated criterion is stateless, i.e.
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local AbstractSequencerCriterion, parent = torch.class('nn.AbstractSequencerCriterion', 'nn.Criterion')

function AbstractSequencerCriterion:__init(criterion, sizeAverage)
   parent.__init(self)
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error(torch.type(self).." shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a ".. torch.type(self) .. ". "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = false
   end
   self.clones = {criterion}
end

function AbstractSequencerCriterion:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion = self.clones[step]
   if not criterion then
      criterion = self.clones[1]:clone()
      self.clones[step] = criterion
   end
   return criterion
end

function AbstractSequencerCriterion:setZeroMask(zeroMask)
   if zeroMask == false then
      for k,stepcriterion in pairs(self.clones) do
         stepcriterion:setZeroMask(zeroMask)
      end
   else
      assert(zeroMask:dim() >= 2, "Expecting dim >= 2 for zeroMask. For example, seqlen x batchsize")
      for step=1,zeroMask:size(1) do
         local stepcriterion = self:getStepCriterion(step)
         stepcriterion:setZeroMask(zeroMask[step])
      end
   end
end

