--[[
The MIT License (MIT)

Copyright (c) 2016 Stéphane Guillitte, Joost van Doorn

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

local SeqGRU, parent = torch.class('nn.SeqGRU', 'nn.Module')

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceGRU stores this many
scalar values:

NTD + 4NTH + 5NH + 6H^2 + 6DH + 7H

Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]

function SeqGRU:__init(inputsize, outputsize)
   parent.__init(self)

   self.inputsize = inputsize
   self.outputsize = outputsize

   self.weight = torch.Tensor(inputsize + outputsize, 3 * outputsize)
   self.gradWeight = torch.Tensor(inputsize + outputsize, 3 * outputsize):zero()
   self.bias = torch.Tensor(3 * outputsize)
   self.gradBias = torch.Tensor(3 * outputsize):zero()
   self:reset()

   self.gates = torch.Tensor() -- This will be (T, N, 3H)
   self.buffer1 = torch.Tensor() -- This will be (N, H)
   self.buffer2 = torch.Tensor() -- This will be (N, H)
   self.buffer3 = torch.Tensor() -- This will be (H,)
   self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)

   self.h0 = torch.Tensor()

   self._remember = 'neither'

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
   self.v2 = true
end

SeqGRU.reset = nn.StepGRU.reset
SeqGRU.zeroMaskState = nn.SeqLSTM.zeroMaskState
SeqGRU.checkZeroMask = nn.SeqLSTM.checkZeroMask

--[[
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function SeqGRU:updateOutput(input)
   self.recompute_backward = true
   assert(torch.isTensor(input))
   local seqlen, batchsize = input:size(1), input:size(2)
   local inputsize, outputsize = self.inputsize, self.outputsize

   self:checkZeroMask(seqlen, batchsize)

   -- remember previous state?
   local remember = self:hasMemory()

   local h0 = self.h0
   if (h0:nElement() ~= batchsize * outputsize) or not remember then
      h0:resize(batchsize, outputsize):zero()
   elseif remember then
      assert(self.output:size(2) == batchsize, 'batch sizes must be the same to remember states')
      h0:copy(self.output[self.output:size(1)])
   end

   local h = self.output
   h:resize(seqlen, batchsize, outputsize):zero()
   self.gates:resize(seqlen, batchsize, 3 * outputsize):zero()

   local prev_h = h0
   if input.nn.StepGRU_updateOutput and not self.forceLua then
      for t = 1, seqlen do
         local cur_x, next_h, gates = input[t], h[t], self.gates[t]
         cur_x.nn.StepGRU_updateOutput(self.weight, self.bias,
                                       gates, cur_x, prev_h,
                                       inputsize, outputsize,
                                       next_h)
         self:zeroMaskState({next_h, gates}, t, cur_x)
         prev_h = next_h
      end
   else
     local bias_expand = self.bias:view(1, 3 * outputsize):expand(batchsize, 3 * outputsize)
     local Wx = self.weight[{{1, inputsize}}]
     local Wh = self.weight[{{inputsize + 1, inputsize + outputsize}}]

     for t = 1, seqlen do
        local cur_x, next_h, cur_gates = input[t], h[t], self.gates[t]

        cur_gates:addmm(bias_expand, cur_x, Wx)
        cur_gates[{{}, {1, 2 * outputsize}}]:addmm(prev_h, Wh[{{}, {1, 2 * outputsize}}])
        cur_gates[{{}, {1, 2 * outputsize}}]:sigmoid()
        local r = cur_gates[{{}, {1, outputsize}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
        local u = cur_gates[{{}, {outputsize + 1, 2 * outputsize}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
        next_h:cmul(r, prev_h) --temporary buffer : r . prev_h
        cur_gates[{{}, {2 * outputsize + 1, 3 * outputsize}}]:addmm(next_h, Wh[{{}, {2 * outputsize + 1, 3 * outputsize}}]) -- hc += Wh * r . prev_h
        local hc = cur_gates[{{}, {2 * outputsize + 1, 3 * outputsize}}]:tanh() --hidden candidate : hc = tanh(Wx * x + Wh * r . prev_h + b)
        next_h:addcmul(hc, -1, u, hc)
        next_h:addcmul(u, prev_h) --next_h = (1-u) . hc + u . prev_h

        self:zeroMaskState({next_h, cur_gates}, t, cur_x)

        prev_h = next_h
     end
  end

  return self.output
end

function SeqGRU:backward(input, gradOutput, scale)
   self.recompute_backward = false
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   local seqlen, batchsize = input:size(1), input:size(2)
   local inputsize, outputsize = self.inputsize, self.outputsize

   local h = self.output

   self.buffer1:resizeAs(self.h0)
   self.gradInput:resizeAs(input):zero()

   local grad_next_h = self.grad_hT or self.buffer1:zero()
   if input.nn.StepGRU_backward and not self.forceLua then
      for t = seqlen, 1, -1 do
         local cur_x, next_h = input[t], h[t]
         local prev_h = (t == 1) and self.h0 or h[t - 1]

         grad_next_h:add(gradOutput[t])
         self:zeroMaskState(grad_next_h, t, cur_x)
         cur_x.nn.StepGRU_backward(self.weight, self.gates[t],
                                  self.gradWeight, self.gradBias, self.grad_a_buffer, self.buffer3,
                                  cur_x, prev_h, grad_next_h,
                                  scale, inputsize, outputsize,
                                  self.gradInput[t], grad_next_h)
      end
   else
      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,outputsize)
      local grad_Wx = self.gradWeight:narrow(1,1,inputsize)
      local grad_Wh = self.gradWeight:narrow(1,inputsize+1,outputsize)
      local grad_b = self.gradBias
      local temp_buffer = self.buffer2:resize(batchsize, outputsize)

      for t = seqlen, 1, -1 do
         local cur_x, next_h = input[t], h[t]
         local prev_h = (t == 1) and self.h0 or h[t - 1]
         grad_next_h:add(gradOutput[t])

         self:zeroMaskState(grad_next_h, t, cur_x)

         local r = self.gates[{t, {}, {1, outputsize}}]
         local u = self.gates[{t, {}, {outputsize + 1, 2 * outputsize}}]
         local hc = self.gates[{t, {}, {2 * outputsize + 1, 3 * outputsize}}]

         local grad_a = self.grad_a_buffer:resize(batchsize, 3 * outputsize):zero()
         local grad_ar = grad_a[{{}, {1, outputsize}}]
         local grad_au = grad_a[{{}, {outputsize + 1, 2 * outputsize}}]
         local grad_ahc = grad_a[{{}, {2 * outputsize + 1, 3 * outputsize}}]

         -- use grad_au as temporary buffer to compute grad_ahc.

         local grad_hc = grad_au:fill(0):addcmul(grad_next_h, -1, u, grad_next_h)
         grad_ahc:fill(1):addcmul(-1, hc,hc):cmul(grad_hc)
         local grad_r = grad_au:fill(0):addmm(grad_ahc, Wh[{{}, {2 * outputsize + 1, 3 * outputsize}}]:t() ):cmul(prev_h)
         grad_ar:fill(1):add(-1, r):cmul(r):cmul(grad_r)

         temp_buffer:fill(0):add(-1, hc):add(prev_h)
         grad_au:fill(1):add(-1, u):cmul(u):cmul(temp_buffer):cmul(grad_next_h)
         self.gradInput[t]:mm(grad_a, Wx:t())
         grad_Wx:addmm(scale, cur_x:t(), grad_a)
         grad_Wh[{{}, {1, 2 * outputsize}}]:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * outputsize}}])

         local grad_a_sum = self.buffer3:resize(outputsize):sum(grad_a, 1)
         grad_b:add(scale, grad_a_sum)
         temp_buffer:fill(0):add(prev_h):cmul(r)
         grad_Wh[{{}, {2 * outputsize + 1, 3 * outputsize}}]:addmm(scale, temp_buffer:t(), grad_ahc)
         grad_next_h:cmul(u)
         grad_next_h:addmm(grad_a[{{}, {1, 2 * outputsize}}], Wh[{{}, {1, 2 * outputsize}}]:t())
         temp_buffer:fill(0):addmm(grad_a[{{}, {2 * outputsize + 1, 3 * outputsize}}], Wh[{{}, {2 * outputsize + 1, 3 * outputsize}}]:t()):cmul(r)
         grad_next_h:add(temp_buffer)
      end
   end

   return self.gradInput
end

function SeqGRU:clearState()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

  self.output:set()
  self.gradInput:set()

  self.zeroMask = nil
end

function SeqGRU:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function SeqGRU:forget()
  self.h0:resize(0)
end

function SeqGRU:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function SeqGRU:type(type, ...)
  self.zeroMask = nil
  self._zeroMask = nil
  self._maskbyte = nil
  self._maskindices = nil
  return parent.type(self, type, ...)
end

SeqGRU.remember = nn.AbstractSequencer.remember
SeqGRU.hasMemory = nn.AbstractSequencer.hasMemory
SeqGRU.training = nn.SeqLSTM.training
SeqGRU.evaluate = nn.SeqLSTM.evaluate
SeqGRU.maskZero = nn.StepGRU.maskZero
SeqGRU.setZeroMask = nn.MaskZero.setZeroMask

function SeqGRU:setStartState(hiddenState)
   self.h0:resizeAs(hiddenState):copy(hiddenState)
end

function SeqGRU:setHiddenState(step, hiddenState)
   if step == 0 then
      self:setStartState(hiddenState)
   else
      error"NotImplemented"
   end
end

function SeqGRU:getHiddenState()
   error"NotImplemented"
end

function SeqGRU:setGradHiddenState()
   error"NotImplemented"
end

function SeqGRU:getGradHiddenState()
   error"NotImplemented"
end

-- used by unit tests
function SeqGRU:toRecGRU()
   assert(not self.weightO)

   local gru = nn.RecGRU(self.inputsize, self.outputsize)
   local stepgru = gru.modules[1]
   stepgru.weight:copy(self.weight)
   stepgru.bias:copy(self.bias)
   stepgru.gradWeight:copy(self.gradWeight)
   stepgru.gradBias:copy(self.gradBias)
   if self.maskzero then
      gru:maskZero()
   end

   return gru
end
