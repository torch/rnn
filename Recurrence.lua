------------------------------------------------------------------------
--[[ Recurrence ]]--
-- A general container for implementing a recurrence.
-- Unlike Recurrent, this module doesn't manage a separate input layer,
-- nor does it have a startModule. Instead for the first step, it
-- just forwards a zero tensor through the recurrent layer (like LSTM).
-- The stepmodule should output Tensor or table : output(t)
-- given input table : {input(t), output(t-1)}
------------------------------------------------------------------------
local _ = require 'moses'
local Recurrence, parent = torch.class('nn.Recurrence', 'nn.AbstractRecurrent')

function Recurrence:__init(stepmodule, outputSize, nInputDim)
   parent.__init(self, stepmodule)

   assert(_.contains({'table','torch.LongStorage','number'}, torch.type(outputSize)), "Unsupported size type")
   self.outputSize = torch.type(outputSize) == 'number' and {outputSize} or outputSize
   -- for table outputs, this is the number of dimensions in the first (left) tensor (depth-first).
   assert(torch.type(nInputDim) == 'number', "Expecting nInputDim number for arg 2")
   self.nInputDim = nInputDim
   assert(torch.isTypeOf(stepmodule, 'nn.Module'), "Expecting stepmodule nn.Module for arg 3")

   -- just so we can know the type of this module
   self.typeTensor = torch.Tensor()
end

-- recursively creates a zero tensor (or table thereof) (or table thereof).
-- This zero Tensor is forwarded as output(t=0).
function Recurrence:recursiveResizeZero(tensor, size, batchSize)
   local isTable = torch.type(size) == 'table'
   if isTable and torch.type(size[1]) ~= 'number' then
      tensor = (torch.type(tensor) == 'table') and tensor or {}
      for k,v in ipairs(size) do
         tensor[k] = self:recursiveResizeZero(tensor[k], v, batchSize)
      end
   elseif torch.type(size) == 'torch.LongStorage'  then
      local size_ = size:totable()
      tensor = torch.isTensor(tensor) and tensor or self.typeTensor.new()
      if batchSize then
         tensor:resize(batchSize, unpack(size_))
      else
         tensor:resize(unpack(size_))
      end
      tensor:zero()
   elseif isTable and torch.type(size[1]) == 'number' then
      tensor = torch.isTensor(tensor) and tensor or self.typeTensor.new()
      if batchSize then
         tensor:resize(batchSize, unpack(size))
      else
         tensor:resize(unpack(size))
      end
      tensor:zero()
   else
      error("Unknown size type : "..torch.type(size))
   end
   return tensor
end

-- get the batch size.
-- When input is a table, we use the first tensor (depth first).
function Recurrence:getBatchSize(input, nInputDim)
   local nInputDim = nInputDim or self.nInputDim
   if torch.type(input) == 'table' then
      return self:getBatchSize(input[1])
   else
      assert(torch.isTensor(input))
      if input:dim() == nInputDim then
         return nil
      elseif input:dim() - 1 == nInputDim then
         return input:size(1)
      else
         error("inconsitent tensor dims "..input:dim())
      end
   end
end

function Recurrence:_updateOutput(input)
   -- output(t-1)
   local prevOutput = self:getHiddenState(self.step-1, input)[1]

   -- output(t) = stepmodule:forward{input(t), output(t-1)}
   local output
   if self.train ~= false then
      local stepmodule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = stepmodule:updateOutput{input, prevOutput}
   else
      -- make a copy of prevOutput to prevent 'output = m:forward(output)' errors
      self._prevOutput = nn.utils.recursiveCopy(self._prevOutput, prevOutput)
      output = self.modules[1]:updateOutput{input, self._prevOutput}
   end

   return output
end

function Recurrence:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local _gradOutput = self:getGradHiddenState(step, input)[1]
   self._gradOutputs[step] = nn.utils.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.utils.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local gradInputTable = stepmodule:updateGradInput({input, self:getHiddenState(step-1)[1]}, gradOutput)

   local _ = require 'moses'
   self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, #gradInputTable))

   return gradInputTable[1]
end

function Recurrence:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local gradOutput = self._gradOutputs[step] or self:getGradHiddenState(step)[1]
   stepmodule:accGradParameters({input, self:getHiddenState(step-1)[1]}, gradOutput, scale)
end

Recurrence.__tostring__ = nn.Decorator.__tostring__

function Recurrence:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput
   if step == 0 then
      if self.startState then
         prevOutput = self.startState
      else
         if input then
            -- first previous output is zeros
            local batchSize = self:getBatchSize(input)
            self.zeroTensor = self:recursiveResizeZero(self.zeroTensor, self.outputSize, batchSize)
         end
         prevOutput = self.outputs[step] or self.zeroTensor
      end
   else
      -- previous output of this module
      prevOutput = self.outputs[step]
   end
   -- call getHiddenState on stepmodule as they may contain AbstractRecurrent instances...
   return {prevOutput, nn.Container.getHiddenState(self, step)}
end

function Recurrence:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.type(hiddenState) == 'table')
   assert(#hiddenState >= 1)

   if step == 0 then
      self:setStartState(hiddenState[1])
   else
      self.outputs[step] = hiddenState[1]
   end

   if hiddenState[2] then
      -- call setHiddenState on stepmodule as they may contain AbstractRecurrent instances...
      nn.Container.setHiddenState(self, step, hiddenState[2])
   end
end

function Recurrence:getGradHiddenState(step, input)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step

   local gradOutput
   if step == self.step-1 and not self.gradOutputs[step] then
      if self.startState then
         local batchSize = self:getBatchSize(input)
         self.zeroTensor = self:recursiveResizeZero(self.zeroTensor, self.outputSize, batchSize)
      end
      gradOutput = self.zeroTensor
   else
      gradOutput = self.gradOutputs[step]
   end
   return {gradOutput, nn.Container.getGradHiddenState(self, step)}
end

function Recurrence:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step

   assert(torch.type(gradHiddenState) == 'table')
   assert(#gradHiddenState >= 1)

   self.gradOutputs[step] = gradHiddenState[1]
   if gradHiddenState[2] then
      nn.Container.setGradHiddenState(self, step, gradHiddenState[2])
   end
end
