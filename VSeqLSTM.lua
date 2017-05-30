local VSeqLSTM, parent = torch.class('nn.VSeqLSTM', 'nn.Module')

function VSeqLSTM:__init(inputsize, outputsize)
   parent.__init(self)

   self.inputsize, self.outputsize, self.hiddensize = inputsize, outputsize, outputsize

   self.weight = torch.Tensor(inputsize+outputsize, 4 * outputsize)
   self.gradWeight = torch.Tensor(inputsize+outputsize, 4 * outputsize)

   self.bias = torch.Tensor(4 * outputsize)
   self.gradBias = torch.Tensor(4 * outputsize):zero()
   self:reset()

   self.cell = torch.Tensor()
   self.buffer = torch.Tensor()
   self.gradInputBuffer = torch.Tensor()
   self.weightBuffer = torch.Tensor()

   self.h0 = torch.Tensor()
   self.c0 = torch.Tensor()

   self._remember = 'neither'

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
   self.v2 = true
end

VSeqLSTM.reset = nn.StepLSTM.reset

--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, ((T1+T2+...+Tn, D)

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function VSeqLSTM:updateOutput(input)
   self.recompute_backward = true
   -- Expect a { torch.Tensor, torch.LongTensor }
   -- where the first tensor is the concatenated array of input values,
   -- following the 'timesteps-first' format
   -- ans the second is the array of decreasing batch sizes
   local linput, sizes
   if torch.isTensor(input) then
      assert(self.__context and self.__context.sizes)
      linput = input
      sizes = self.__context.sizes
   elseif type(input) == 'table' and input[1] and input[2] then
      linput = input[1]
      sizes = input[2]
   else
      error('Cannot recognize input')
   end
   local batchsize = sizes[1]
   local inputsize, outputsize = self.inputsize, self.outputsize


   -- remember previous state?
   local remember = self:hasMemory()

   local c0 = self.c0
   if (c0:nElement() ~= batchsize * outputsize) or not remember then
      c0:resize(batchsize, outputsize):zero()
   elseif remember then
      assert(self.cell:size(2) == batchsize, 'batch sizes must be constant to remember states')
      c0:copy(self.cell[self.cell:size(1)])
   end

   local h0 = self.h0
   if (h0:nElement() ~= batchsize * outputsize) or not remember then
      h0:resize(batchsize, outputsize):zero()
   elseif remember then
      assert(self.output:size(2) == batchsize, 'batch sizes must be the same to remember states')
      h0:copy(self.output[self.output:size(1)])
   end

   local h, c = self.output, self.cell

   linput.THRNN.LSTM_updateOutput(
             linput:cdata(),
             c0:cdata(),
             h0:cdata(),
             sizes:cdata(),
             self.output:cdata(),
             self.weight:cdata(),
             self.bias:cdata(),
             self.buffer:cdata(),
             self.inputsize,
             self.outputsize,
             1,
             0,
             0)


   return self.output
end

function VSeqLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   local linput, sizes
   if torch.isTensor(input) then
      assert(self.__context and self.__context.sizes)
      linput = input
      sizes = self.__context.sizes
   elseif type(input) == 'table' and input[1] and input[2] then
      linput = input[1]
      sizes = input[2]
   else
      error('Cannot recognize input')
   end
   local batchsize = sizes[1]
   local inputsize, outputsize = self.inputsize, self.outputsize

   if not self.grad_hT then
      self.gradInputH = self.gradInputH or self.h0.new()
      self.gradInputH:resize(self.h0:size()):zero()
   else
      self.gradInputH = self.grad_hT
   end

   if not self.grad_cT then
      self.gradInputC = self.gradInputC or self.c0.new()
      self.gradInputC:resize(self.c0:size()):zero()
   else
      self.gradInputC = self.grad_cT
   end


   linput.THRNN.LSTM_backward(
             linput:cdata(),
             self.c0:cdata(),
             self.h0:cdata(),
             sizes:cdata(),
             gradOutput:cdata(),
             self.gradInput:cdata(),
             self.gradInputH:cdata(),
             self.gradInputC:cdata(),
             self.weight:cdata(),
             self.bias:cdata(),
             self.buffer:cdata(),
             self.weightBuffer:cdata(),
             self.gradInputBuffer:cdata(),
             self.gradWeight:cdata(),
             self.gradBias:cdata(),
             self.output:cdata(),
             scale,
             0,
             inputsize,
             outputsize,
             1,
             0)

   return self.gradInput
end

function VSeqLSTM:clearState()
   self.cell:set()
   self.buffer:set()
   self.weightBuffer:set()
   self.gradInputBuffer:set()
   self.c0:set()
   self.h0:set()

   self.output:set()
   self.gradInput:set()
   self.grad_hidden = nil
   self.hidden = nil

   self.zeroMask = nil
end

function VSeqLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function VSeqLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function VSeqLSTM:forget()
   self.c0:resize(0)
   self.h0:resize(0)
end

function VSeqLSTM:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
VSeqLSTM.remember = nn.AbstractSequencer.remember
VSeqLSTM.hasMemory = nn.AbstractSequencer.hasMemory

function VSeqLSTM:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function VSeqLSTM:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

VSeqLSTM.maskZero = nn.StepLSTM.maskZero
VSeqLSTM.setZeroMask = nn.MaskZero.setZeroMask
VSeqLSTM.__tostring__ = nn.StepLSTM.__tostring__

function VSeqLSTM:parameters()
   return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
end

function VSeqLSTM:setStartState(hiddenState)
   self.h0:resizeAs(hiddenState[1]):copy(hiddenState[1])
   self.c0:resizeAs(hiddenState[2]):copy(hiddenState[2])
end

function VSeqLSTM:setHiddenState(step, hiddenState)
   if step == 0 then
      self:setStartState(hiddenState)
   else
      error"NotImplemented"
   end
end

function VSeqLSTM:getHiddenState()
   error"NotImplemented"
end

function VSeqLSTM:setGradHiddenState()
   error"NotImplemented"
end

function VSeqLSTM:getGradHiddenState()
   error"NotImplemented"
end

