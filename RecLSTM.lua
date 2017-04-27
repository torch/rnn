local RecLSTM, parent = torch.class('nn.RecLSTM', 'nn.AbstractRecurrent')

function RecLSTM:__init(inputsize, hiddensize, outputsize)
   local stepmodule = nn.StepLSTM(inputsize, hiddensize, outputsize)
   parent.__init(self, stepmodule)
   -- let StepLSTM initialize inputsize, hiddensize, outputsize
   self.inputsize = self.modules[1].inputsize
   self.hiddensize = self.modules[1].hiddensize
   self.outputsize = self.modules[1].outputsize

   self.prev_h0 = torch.Tensor()
   self.prev_c0 = torch.Tensor()

   self.cells = {}
   self.gradCells = {}
end

function RecLSTM:maskZero()
   assert(torch.isTypeOf(self.modules[1], 'nn.StepLSTM'))
   for i,stepmodule in pairs(self.sharedClones) do
      stepmodule.maskzero = true
   end
   self.modules[1].maskzero = true
   return self
end

function RecLSTM:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput, prevCell
   if step == 0 then
      prevOutput = self.userPrevOutput or self.outputs[step] or self.prev_h0
      prevCell = self.userPrevCell or self.cells[step] or self.prev_c0
      if input then
         if input:dim() == 2 then
            self.prev_h0:resize(input:size(1), self.outputsize):zero()
            self.prev_c0:resize(input:size(1), self.hiddensize):zero()
         else
            self.prev_h0:resize(self.outputsize):zero()
            self.prev_c0:resize(self.hiddensize):zero()
         end
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

   -- previous output of this module
   self.outputs[step] = hiddenState[1]
   self.cells[step] = hiddenState[2]
end

------------------------- forward backward -----------------------------
function RecLSTM:updateOutput(input)
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

   self.outputs[self.step] = output
   self.cells[self.step] = cell

   self.output = output
   self.cell = cell

   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   -- note that we don't return the cell, just the output
   return self.output
end

function RecLSTM:getGradHiddenState(step)
   self.gradOutputs = self.gradOutputs or {}
   self.gradCells = self.gradCells or {}
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput, gradCell
   if step == self.step-1 then
      gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.prev_h0
      gradCell = self.userNextGradCell or self.gradCells[step] or self.prev_c0
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

function RecLSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local gradHiddenState = self:getGradHiddenState(step)
   local _gradOutput, gradCell = gradHiddenState[1], gradHiddenState[2]
   assert(_gradOutput and gradCell)

   self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)

   local gradInputTable = stepmodule:updateGradInput(inputTable, {gradOutput, gradCell})

   local _ = require 'moses'
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
   self.prev_c0:set()
   self.prev_h0:set()
   if self.userPrevOutput then self.userPrevOutput:set() end
   if self.userPrevCell then self.userPrevCell:set() end
   if self.userGradPrevOutput then self.userGradPrevOutput:set() end
   if self.userGradPrevCell then self.userGradPrevCell:set() end
   return parent.clearState(self)
end

function RecLSTM:type(type, ...)
   if type then
      self:forget()
      self:clearState()
   end
   return parent.type(self, type, ...)
end