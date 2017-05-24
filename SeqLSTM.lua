--[[
The MIT License (MIT)

Copyright (c) 2016 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--]]

--[[
Thank you Justin for this awesome super fast code:
 * https://github.com/jcjohnson/torch-rnn

If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SeqLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

N : batchsize; T : seqlen; D : inputsize; H : outputsize

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]
local SeqLSTM, parent = torch.class('nn.SeqLSTM', 'nn.Module')

function SeqLSTM:__init(inputsize, hiddensize, outputsize)
   parent.__init(self)

   if hiddensize and outputsize then
      -- implements LSTMP
      self.weightO = torch.Tensor(hiddensize, outputsize)
      self.gradWeightO = torch.Tensor(hiddensize, outputsize)
   else
      -- implements LSTM
      assert(inputsize and hiddensize and not outputsize)
      outputsize = hiddensize
   end
   self.inputsize, self.hiddensize, self.outputsize = inputsize, hiddensize, outputsize

   self.weight = torch.Tensor(inputsize+outputsize, 4 * hiddensize)
   self.gradWeight = torch.Tensor(inputsize+outputsize, 4 * hiddensize)

   self.bias = torch.Tensor(4 * hiddensize)
   self.gradBias = torch.Tensor(4 * hiddensize):zero()
   self:reset()

   self.cell = torch.Tensor()    -- This will be  (T, N, H)
   self.gates = torch.Tensor()   -- This will be (T, N, 4H)
   self.buffer1 = torch.Tensor() -- This will be (N, H)
   self.buffer2 = torch.Tensor() -- This will be (N, H)
   self.buffer3 = torch.Tensor() -- This will be (1, 4H)
   self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

   self.h0 = torch.Tensor()
   self.c0 = torch.Tensor()

   self._remember = 'neither'

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
   self.v2 = true
end

SeqLSTM.reset = nn.StepLSTM.reset

function SeqLSTM:zeroMaskState(state, step, cur_x)
   if self.maskzero and self.zeroMask ~= false then
      if self.v2 then
         assert(self.zeroMask ~= nil, torch.type(self).." expecting zeroMask tensor or false")
         nn.utils.recursiveZeroMask(state, self.zeroMask[step])
      else -- backwards compat
         self.zeroMask = nn.utils.getZeroMaskBatch(cur_x, self.zeroMask)
         nn.utils.recursiveZeroMask(state, self.zeroMask)
      end
   end
end

function SeqLSTM:checkZeroMask(seqlen, batchsize)
   if self.maskzero and self.v2 and self.zeroMask ~= false then
      if not torch.isTensor(self.zeroMask) then
         error(torch.type(self).." expecting previous call to setZeroMask(zeroMask) with maskzero=true")
      end
      if (self.zeroMask:size(1) ~= seqlen) or (self.zeroMask:size(2) ~= batchsize) then
         error(torch.type(self).." expecting zeroMask of size seqlen x batchsize, got "
            ..self.zeroMask:size(1).." x "..self.zeroMask:size(2).." instead of "..seqlen.." x "..batchsize )
      end
   end
end

--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function SeqLSTM:updateOutput(input)
   self.recompute_backward = true
   assert(torch.isTensor(input))
   local seqlen, batchsize = input:size(1), input:size(2)
   local inputsize, hiddensize, outputsize = self.inputsize, self.hiddensize, self.outputsize

   self:checkZeroMask(seqlen, batchsize)

   -- remember previous state?
   local remember = self:hasMemory()

   local c0 = self.c0
   if (c0:nElement() ~= batchsize * hiddensize) or not remember then
      c0:resize(batchsize, hiddensize):zero()
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
   h:resize(seqlen, batchsize, outputsize)
   c:resize(seqlen, batchsize, hiddensize)

   local nElement = self.gates:nElement()
   self.gates:resize(seqlen, batchsize, 4 * hiddensize)
   if nElement ~= seqlen * batchsize * 4 * hiddensize then
      self.gates:zero()
   end

   local prev_h, prev_c = h0, c0
   if input.nn and input.nn.StepLSTM_updateOutput and not self.forceLua then
      for t = 1, seqlen do
         local cur_x, next_h, next_c, gates = input[t], h[t], c[t], self.gates[t]

         if self.weightO then
            self._hidden = self._hidden or next_h.new()
            self._hidden:resize(seqlen, batchsize, hiddensize)
            cur_x.nn.StepLSTM_updateOutput(self.weight, self.bias,
                                           gates, cur_x, prev_h, prev_c,
                                           inputsize, hiddensize, outputsize,
                                           self._hidden[t], next_c, self.weightO, next_h)
         else
            cur_x.nn.StepLSTM_updateOutput(self.weight, self.bias,
                                           gates, cur_x, prev_h, prev_c,
                                           inputsize, hiddensize, outputsize,
                                           next_h, next_c)
         end

         self:zeroMaskState({next_h, next_c, gates}, t, cur_x)

         prev_h, prev_c = next_h, next_c
      end
   else
      local bias_expand = self.bias:view(1, 4 * hiddensize):expand(batchsize, 4 * hiddensize)
      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,outputsize)

      for t = 1, seqlen do
         local cur_x, next_h, next_c, cur_gates = input[t], h[t], c[t], self.gates[t]

         cur_gates:addmm(bias_expand, cur_x, Wx)
         cur_gates:addmm(prev_h, Wh)
         cur_gates[{{}, {1, 3 * hiddensize}}]:sigmoid()
         cur_gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]:tanh()
         local i = cur_gates[{{}, {1, hiddensize}}] -- input gate
         local f = cur_gates[{{}, {hiddensize + 1, 2 * hiddensize}}] -- forget gate
         local o = cur_gates[{{}, {2 * hiddensize + 1, 3 * hiddensize}}] -- output gate
         local g = cur_gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}] -- input transform
         next_h:cmul(i, g)
         next_c:cmul(f, prev_c):add(next_h)
         next_h:tanh(next_c):cmul(o)

         if self.weightO then -- LSTMP
            self._hidden = self._hidden or next_h.new()
            self._hidden:resize(seqlen, batchsize, self.hiddensize)

            self._hidden[t]:copy(next_h)
            next_h:resize(batchsize,self.outputsize)
            next_h:mm(self._hidden[t], self.weightO)
         end

         self:zeroMaskState({next_h, next_c, cur_gates}, t, cur_x)

         prev_h, prev_c = next_h, next_c
      end

   end

   return self.output
end

function SeqLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   local seqlen, batchsize = input:size(1), input:size(2)
   local inputsize, hiddensize, outputsize = self.inputsize, self.hiddensize, self.outputsize

   local h, c = self.output, self.cell

   self.buffer1:resizeAs(self.h0)
   self.buffer2:resizeAs(self.c0)
   self.gradInput:resizeAs(input)

   local grad_next_h = self.grad_hT or self.buffer1:zero()
   local grad_next_c = self.grad_cT or self.buffer2:zero()
   if input.nn and input.nn.StepLSTM_backward and not self.forceLua then
      for t = seqlen, 1, -1 do
         local cur_x, next_h, next_c = input[t], h[t], c[t]
         local prev_h, prev_c
         if t == 1 then
            prev_h, prev_c = self.h0, self.c0
         else
            prev_h, prev_c = h[t - 1], c[t - 1]
         end
         grad_next_h:add(gradOutput[t])

         self:zeroMaskState({grad_next_h, grad_next_c}, t, cur_x)

         if self.weightO then
            self.grad_hidden = self.grad_hidden or cur_x.new()
            cur_x.nn.StepLSTM_backward(self.weight, self.gates[t], self.gradWeight, self.gradBias,
                                       self.grad_a_buffer, self.buffer3,
                                       cur_x, prev_h, prev_c, next_c,
                                       grad_next_h, grad_next_c,
                                       scale, inputsize, hiddensize, outputsize,
                                       self.gradInput[t], grad_next_h, grad_next_c,
                                       self.weightO, self._hidden[t], self.gradWeightO, self.grad_hidden)
         else
            cur_x.nn.StepLSTM_backward(self.weight, self.gates[t], self.gradWeight, self.gradBias,
                                       self.grad_a_buffer, self.buffer3,
                                       cur_x, prev_h, prev_c, next_c,
                                       grad_next_h, grad_next_c,
                                       scale, inputsize, hiddensize, outputsize,
                                       self.gradInput[t], grad_next_h, grad_next_c)
         end
      end
   else
      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,outputsize)
      local grad_Wx = self.gradWeight:narrow(1,1,inputsize)
      local grad_Wh = self.gradWeight:narrow(1,inputsize+1,outputsize)
      local grad_b = self.gradBias

      for t = seqlen, 1, -1 do
         local cur_x, next_h, next_c = input[t], h[t], c[t]
         local prev_h, prev_c = nil, nil
         if t == 1 then
            prev_h, prev_c = self.h0, self.c0
         else
            prev_h, prev_c = h[t - 1], c[t - 1]
         end
         grad_next_h:add(gradOutput[t])

         local cur_x = input[t]

         self:zeroMaskState({grad_next_h, grad_next_c}, t, cur_x)

         if self.weightO then -- LSTMP
            self.buffer3:resizeAs(grad_next_h):copy(grad_next_h)

            self.gradWeightO:addmm(scale, self._hidden[t]:t(), grad_next_h)
            grad_next_h:resize(batchsize, hiddensize)
            grad_next_h:mm(self.buffer3, self.weightO:t())
         end

         local i = self.gates[{t, {}, {1, hiddensize}}]
         local f = self.gates[{t, {}, {hiddensize + 1, 2 * hiddensize}}]
         local o = self.gates[{t, {}, {2 * hiddensize + 1, 3 * hiddensize}}]
         local g = self.gates[{t, {}, {3 * hiddensize + 1, 4 * hiddensize}}]

         local grad_a = self.grad_a_buffer:resize(batchsize, 4 * hiddensize):zero()
         local grad_ai = grad_a[{{}, {1, hiddensize}}]
         local grad_af = grad_a[{{}, {hiddensize + 1, 2 * hiddensize}}]
         local grad_ao = grad_a[{{}, {2 * hiddensize + 1, 3 * hiddensize}}]
         local grad_ag = grad_a[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]

         -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
         -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
         -- to compute grad_ao; the other values can be overwritten after we compute
         -- grad_next_c
         local tanh_next_c = grad_ai:tanh(next_c)
         local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
         local my_grad_next_c = grad_ao
         my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
         grad_next_c:add(my_grad_next_c)

         -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
         -- that we can overwrite it.
         grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

         -- Use grad_ai as a temporary buffer for computing grad_ag
         local g2 = grad_ai:cmul(g, g)
         grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

         -- We don't need any temporary storage for these so do them last
         grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
         grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)

         self.gradInput[t]:mm(grad_a, Wx:t())
         grad_Wx:addmm(scale, cur_x:t(), grad_a)
         grad_Wh:addmm(scale, prev_h:t(), grad_a)
         local grad_a_sum = self.buffer3:resize(1, 4 * hiddensize):sum(grad_a, 1)
         grad_b:add(scale, grad_a_sum)

         grad_next_h:resize(batchsize, outputsize)
         grad_next_h:mm(grad_a, Wh:t())
         grad_next_c:cmul(f)
      end
   end

   return self.gradInput
end

function SeqLSTM:clearState()
   self.cell:set()
   self.gates:set()
   self.buffer1:set()
   self.buffer2:set()
   self.buffer3:set()
   self.grad_a_buffer:set()
   self.c0:set()
   self.h0:set()

   self.output:set()
   self.gradInput:set()
   self.grad_hidden = nil
   self.hidden = nil

   self.zeroMask = nil
end

function SeqLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function SeqLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function SeqLSTM:forget()
   self.c0:resize(0)
   self.h0:resize(0)
end

function SeqLSTM:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
SeqLSTM.remember = nn.AbstractSequencer.remember
SeqLSTM.hasMemory = nn.AbstractSequencer.hasMemory

function SeqLSTM:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function SeqLSTM:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

SeqLSTM.maskZero = nn.StepLSTM.maskZero
SeqLSTM.setZeroMask = nn.MaskZero.setZeroMask
SeqLSTM.__tostring__ = nn.StepLSTM.__tostring__

function SeqLSTM:parameters()
   return {self.weight, self.bias, self.weightO}, {self.gradWeight, self.gradBias, self.gradWeightO}
end

function SeqLSTM:setStartState(hiddenState)
   self.h0:resizeAs(hiddenState[1]):copy(hiddenState[1])
   self.c0:resizeAs(hiddenState[2]):copy(hiddenState[2])
end

function SeqLSTM:setHiddenState(step, hiddenState)
   if step == 0 then
      self:setStartState(hiddenState)
   else
      error"NotImplemented"
   end
end

function SeqLSTM:getHiddenState()
   error"NotImplemented"
end

function SeqLSTM:setGradHiddenState()
   error"NotImplemented"
end

function SeqLSTM:getGradHiddenState()
   error"NotImplemented"
end


-- for sharedClone
SeqLSTM.dpnn_parameters = {'weight', 'bias', 'weightO'}
SeqLSTM.dpnn_gradParameters = {'gradWeight', 'gradBias', 'gradWeightO'}

-- used by unit tests
function SeqLSTM:toRecLSTM()
   assert(not self.weightO)

   local lstm = nn.RecLSTM(self.inputsize, self.outputsize)
   local steplstm = lstm.modules[1]
   steplstm.weight:copy(self.weight)
   steplstm.bias:copy(self.bias)
   steplstm.gradWeight:copy(self.gradWeight)
   steplstm.gradBias:copy(self.gradBias)
   if self.maskzero then
      lstm:maskZero()
   end

   return lstm
end