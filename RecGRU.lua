local _ = require 'moses'
local RecGRU, parent = torch.class('nn.RecGRU', 'nn.AbstractRecurrent')

function RecGRU:__init(inputsize, outputsize)
   local stepmodule = nn.StepGRU(inputsize, outputsize)
   parent.__init(self, stepmodule)
   self.inputsize = inputsize
   self.outputsize = outputsize

   self.zeroOutput = torch.Tensor()
end

function RecGRU:maskZero(v1)
   assert(torch.isTypeOf(self.modules[1], 'nn.StepGRU'))
   for i,stepmodule in pairs(self.sharedClones) do
      stepmodule:maskZero(v1)
   end
   self.modules[1]:maskZero(v1)
   return self
end

------------------------- forward backward -----------------------------
function RecGRU:_updateOutput(input)
   assert(input:dim() == 2, "RecGRU expecting batchsize x inputsize tensor (Only supports batchmode)")
   local prevOutput = self:getHiddenState(self.step-1, input)

   -- output(t) = gru{input(t), output(t-1)}
   local output
   if self.train ~= false then
      local stepmodule = self:getStepModule(self.step)
      output = stepmodule:updateOutput({input, prevOutput})
   else
      self._prevOutput = self._prevOutput or prevOutput.new()
      self._prevOutput:resizeAs(prevOutput):copy(prevOutput)
      output = self.modules[1]:updateOutput({input, self._prevOutput})
   end

   return output
end

function RecGRU:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local _gradOutput = assert(self:getGradHiddenState(step, input))

   self._gradOutputs[step] = self._gradOutputs[step] or _gradOutput.new()
   self._gradOutputs[step]:resizeAs(_gradOutput)
   self._gradOutputs[step]:add(_gradOutput, gradOutput)
   gradOutput = self._gradOutputs[step]

   local inputTable = {input, self:getHiddenState(step-1)}
   local gradInputTable = stepmodule:updateGradInput(inputTable, gradOutput)

   self:setGradHiddenState(step-1, gradInputTable[2])

   return gradInputTable[1]
end

function RecGRU:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local inputTable = {input, self:getHiddenState(step-1)}
   local gradOutput = self._gradOutputs[step] or self:getGradHiddenState(step)
   stepmodule:accGradParameters(inputTable, gradOutput, scale)
end

function RecGRU:clearState()
   self.startState = nil
   self.zeroOutput:set()
   return parent.clearState(self)
end

function RecGRU:type(type, ...)
   if type then
      self:forget()
      self:clearState()
   end
   return parent.type(self, type, ...)
end


function RecGRU:initZeroTensor(input)
   if input then
      if input:dim() == 2 then
         self.zeroOutput:resize(input:size(1), self.outputsize):zero()
      else
         self.zeroOutput:resize(self.outputsize):zero()
      end
   end
end

function RecGRU:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput
   if step == 0 then
      if self.startState then
         prevOutput = self.startState
         if input and input:dim() == 2 then
            assert(prevOutput:size(2) == self.outputsize)
            assert(prevOutput:size(1) == input:size(1))
         end
      else
         prevOutput = self.zeroOutput
         self:initZeroTensor(input)
      end
   else
      -- previous output of this module
      prevOutput = self.outputs[step]
   end
   return prevOutput
end

function RecGRU:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.isTensor(hiddenState))

   if step == 0 then
      -- this hack bipasses the fact that Sequencer calls forget when remember is false
      -- which makes it impossible to use self.outputs to set the h[0] (it is forgotten)
      self:setStartState(hiddenState)
   else
      -- previous output of this module
      self.outputs[step] = hiddenState
   end
end

function RecGRU:getGradHiddenState(step, input)
   self.gradOutputs = self.gradOutputs or {}
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput
   if step == self.step-1 then
      if self.startState and not self.gradOutputs[step] then
         self:initZeroTensor(input)
      end
      gradOutput = self.gradOutputs[step] or self.zeroOutput
   else
      gradOutput = self.gradOutputs[step]
   end
   return gradOutput
end

function RecGRU:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   assert(torch.isTensor(gradHiddenState))

   self.gradOutputs[step] = gradHiddenState
end

function RecGRU:__tostring__()
   if self.weightO then
       return self.__typename .. string.format("(%d -> %d -> %d)", self.inputsize, self.hiddensize, self.outputsize)
   else
       return self.__typename .. string.format("(%d -> %d)", self.inputsize, self.outputsize)
   end
end