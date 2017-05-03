local _ = require 'moses'
local RecLSTM, parent = torch.class('nn.RecLSTM', 'nn.AbstractRecurrent')

function RecLSTM:__init(inputsize, hiddensize, outputsize)
   local stepmodule = nn.StepLSTM(inputsize, hiddensize, outputsize)
   parent.__init(self, stepmodule)
   -- let StepLSTM initialize inputsize, hiddensize, outputsize
   self.inputsize = self.modules[1].inputsize
   self.hiddensize = self.modules[1].hiddensize
   self.outputsize = self.modules[1].outputsize

   self.cells = {}
   self.gradCells = {}

   self.zeroOutput = torch.Tensor()
   self.zeroCell = torch.Tensor()
end

function RecLSTM:maskZero(v1)
   assert(torch.isTypeOf(self.modules[1], 'nn.StepLSTM'))
   for i,stepmodule in pairs(self.sharedClones) do
      stepmodule:maskZero(v1)
   end
   self.modules[1]:maskZero(v1)
   return self
end

------------------------- forward backward -----------------------------
function RecLSTM:_updateOutput(input)
   assert(input:dim() == 2, "RecLSTM expecting batchsize x inputsize tensor (Only supports batchmode)")
   local prevOutput, prevCell = unpack(self:getHiddenState(self.step-1, input))

   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local stepmodule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(stepmodule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.modules[1]:updateOutput{input, prevOutput, prevCell})
   end

   self.cells[self.step] = cell
   self.cell = cell
   -- note that we don't return the cell, just the output
   return output
end

function RecLSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local gradHiddenState = self:getGradHiddenState(step, input)
   local _gradOutput, gradCell = gradHiddenState[1], gradHiddenState[2]
   assert(_gradOutput and gradCell)

   self._gradOutputs[step] = self._gradOutputs[step] or _gradOutput.new()
   self._gradOutputs[step]:resizeAs(_gradOutput)
   self._gradOutputs[step]:add(_gradOutput, gradOutput)
   gradOutput = self._gradOutputs[step]

   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)

   local gradInputTable = stepmodule:updateGradInput(inputTable, {gradOutput, gradCell})

   self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, 3))

   return gradInputTable[1]
end

function RecLSTM:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)
   local gradOutputTable = self:getGradHiddenState(step)
   gradOutputTable[1] = self._gradOutputs[step] or gradOutputTable[1]
   stepmodule:accGradParameters(inputTable, gradOutputTable, scale)
end

function RecLSTM:clearState()
   self.startState = nil
   self.zeroCell:set()
   self.zeroOutput:set()
   return parent.clearState(self)
end

function RecLSTM:type(type, ...)
   if type then
      self:forget()
      self:clearState()
   end
   return parent.type(self, type, ...)
end


function RecLSTM:initZeroTensor(input)
   if input then
      if input:dim() == 2 then
         self.zeroOutput:resize(input:size(1), self.outputsize):zero()
         self.zeroCell:resize(input:size(1), self.hiddensize):zero()
      else
         self.zeroOutput:resize(self.outputsize):zero()
         self.zeroCell:resize(self.hiddensize):zero()
      end
   end
end

function RecLSTM:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput, prevCell
   if step == 0 then
      if self.startState then
         prevOutput, prevCell = self.startState[1], self.startState[2]
         if input and input:dim() == 2 then
            assert(prevOutput:size(2) == self.outputsize)
            assert(prevCell:size(2) == self.hiddensize)
            assert(prevOutput:size(1) == input:size(1))
            assert(prevCell:size(1) == input:size(1))
         end
      else
         prevOutput = self.zeroOutput
         prevCell = self.zeroCell
         self:initZeroTensor(input)
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[step]
      prevCell = self.cells[step]
   end
   return {prevOutput, prevCell}
end

function RecLSTM:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.type(hiddenState) == 'table')
   assert(#hiddenState == 2)

   if step == 0 then
      -- this hack bipasses the fact that Sequencer calls forget when remember is false
      -- which makes it impossible to use self.outputs to set the h[0] (it is forgotten)
      self:setStartState(hiddenState)
   else
      -- previous output of this module
      self.outputs[step] = hiddenState[1]
      self.cells[step] = hiddenState[2]
   end
end

function RecLSTM:getGradHiddenState(step, input)
   self.gradOutputs = self.gradOutputs or {}
   self.gradCells = self.gradCells or {}
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput, gradCell
   if step == self.step-1 then
      if self.startState and not (self.gradOutputs[step] and self.gradCells[input]) then
         self:initZeroTensor(input)
      end
      gradOutput = self.gradOutputs[step] or self.zeroOutput
      gradCell = self.gradCells[step] or self.zeroCell
   else
      gradOutput = self.gradOutputs[step]
      gradCell = self.gradCells[step]
   end
   return {gradOutput, gradCell}
end

function RecLSTM:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   assert(torch.type(gradHiddenState) == 'table')
   assert(#gradHiddenState == 2)

   self.gradOutputs[step] = gradHiddenState[1]
   self.gradCells[step] = gradHiddenState[2]
end

function RecLSTM:__tostring__()
   if self.weightO then
       return self.__typename .. string.format("(%d -> %d -> %d)", self.inputsize, self.hiddensize, self.outputsize)
   else
       return self.__typename .. string.format("(%d -> %d)", self.inputsize, self.outputsize)
   end
end