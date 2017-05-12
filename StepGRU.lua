-- StepGRU is a step-wise module that can be used inside Recurrence to implement an GRU.
-- That is, the StepGRU efficiently implements a single GRU time-step.
-- Its efficient because it doesn't use any internal modules; it calls BLAS directly.
-- StepGRU is based on SeqGRU.
-- Input : {input[t], hidden[t-1]}
-- Output: hidden[t]
local StepGRU, parent = torch.class('nn.StepGRU', 'nn.Module')

function StepGRU:__init(inputsize, outputsize)
   parent.__init(self)
   self.inputsize, self.outputsize = inputsize, outputsize

   self.weight = torch.Tensor(inputsize+outputsize, 3 * outputsize)
   self.gradWeight = torch.Tensor(inputsize+outputsize, 3 * outputsize)

   self.bias = torch.Tensor(3 * outputsize)
   self.gradBias = torch.Tensor(3 * outputsize):zero()
   self:reset()

   self.gates = torch.Tensor() -- batchsize x 4*outputsize

   self.gradInput = {torch.Tensor(), torch.Tensor()}

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
   self.v2 = true
end

function StepGRU:reset(std)
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   self.weight:normal(0, std or (1.0 / math.sqrt(self.outputsize + self.inputsize)))
   return self
end

function StepGRU:updateOutput(input)
   self.recompute_backward = true
   local cur_x, prev_h, next_h = input[1], input[2], self.output
   if cur_x.nn and cur_x.nn.StepGRU_updateOutput and not self.forceLua then
      cur_x.nn.StepGRU_updateOutput(self.weight, self.bias, self.gates,
                                    cur_x, prev_h,
                                    self.inputsize, self.outputsize,
                                    next_h)
   else
      assert(torch.isTensor(prev_h))
      local batchsize, inputsize, outputsize = cur_x:size(1), cur_x:size(2), self.outputsize
      assert(inputsize == self.inputsize)

      local bias_expand = self.bias:view(1, 3 * outputsize):expand(batchsize, 3 * outputsize)
      local Wx = self.weight:narrow(1, 1, inputsize)
      local Wh = self.weight:narrow(1, inputsize + 1, self.outputsize)

      next_h:resize(batchsize, outputsize)
      self.gates:resize(batchsize, 3 * outputsize):zero()
      local gates = self.gates

      gates:addmm(bias_expand, cur_x, Wx)
      local sub_gates = gates:narrow(2, 1, 2 * outputsize)
      sub_gates:addmm(prev_h, Wh:narrow(2, 1, 2 * outputsize))
      sub_gates:sigmoid()
      local reset_gate = gates:narrow(2, 1, outputsize) -- r = sig(Wx * x + Wh * prev_h + b)
      local update_gate = gates:narrow(2, outputsize + 1, outputsize) -- u = sig(Wx * x + Wh * prev_h + b)
      local hidden_candidate = gates:narrow(2, 2 * outputsize + 1, outputsize) -- hc = tanh(Wx * x + Wh * r . prev_h + b)

      next_h:cmul(reset_gate, prev_h) --temporary buffer : r . prev_h

      hidden_candidate:addmm(next_h, Wh:narrow(2, 2 * outputsize + 1, outputsize)) -- hc += Wh * r . prev_h
      hidden_candidate:tanh() -- hc = tanh(Wx * x + Wh * r . prev_h + b)
      next_h:addcmul(hidden_candidate, -1, update_gate, hidden_candidate) -- (1-u) . hc = hc - (u . hc)
      next_h:addcmul(update_gate, prev_h) --next_h = (1-u) . hc + u . prev_h
   end

   if self.maskzero and self.zeroMask ~= false then
      if self.v2 then
         assert(self.zeroMask ~= nil, torch.type(self).." expecting zeroMask tensor or false")
      else -- backwards compat
         self.zeroMask = nn.utils.getZeroMaskBatch(cur_x, self.zeroMask)
      end
      -- zero masked outputs and gates
      nn.utils.recursiveZeroMask({next_h, self.gates}, self.zeroMask)
   end

   return self.output
end

function StepGRU:backward(input, gradOutput, scale)
   self.recompute_backward = false
   local cur_x, prev_h = input[1], input[2]
   local grad_next_h = gradOutput
   local grad_cur_x, grad_prev_h = self.gradInput[1], self.gradInput[2]
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   --
   local grad_gates = torch.getBuffer('StepGRU', 'grad_gates', self.gates) -- batchsize x 3*outputsize
   local buffer = torch.getBuffer('StepGRU', 'buffer', self.gates) -- 1 x 3*outputsize

   if self.maskzero and self.zeroMask ~= false then
      -- zero masked gradOutput
      nn.utils.recursiveZeroMask(grad_next_h, self.zeroMask)
   end

   if cur_x.nn and cur_x.nn.StepGRU_backward and not self.forceLua then
      cur_x.nn.StepGRU_backward(self.weight, self.gates,
                                self.gradWeight, self.gradBias, grad_gates, buffer,
                                cur_x, prev_h, grad_next_h,
                                scale, self.inputsize, self.outputsize,
                                grad_cur_x, grad_prev_h)
   else
      local batchsize, inputsize, outputsize = cur_x:size(1), cur_x:size(2), self.outputsize
      assert(inputsize == self.inputsize)

      grad_cur_x:resize(batchsize, inputsize)
      grad_prev_h:resize(batchsize, outputsize)

      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,outputsize)
      local grad_Wx = self.gradWeight:narrow(1,1,inputsize)
      local grad_Wh = self.gradWeight:narrow(1,inputsize+1,outputsize)
      local grad_b = self.gradBias

      local gates = self.gates
      local reset_gate = gates:narrow(2, 1, outputsize)
      local update_gate = gates:narrow(2, outputsize + 1, outputsize)
      local hidden_candidate = gates:narrow(2, 2 * outputsize + 1, outputsize)

      grad_gates:resize(batchsize, 3 * outputsize):zero()
      local grad_reset_gate = grad_gates:narrow(2, 1, outputsize)
      local grad_update_gate = grad_gates:narrow(2, outputsize + 1, outputsize)
      local grad_hidden_candidate = grad_gates:narrow(2, 2 * outputsize + 1, outputsize)

      -- use grad_update_gate as temporary buffer to compute grad_hidden_candidate and grad_reset_gate
      grad_update_gate:fill(0):addcmul(grad_next_h, -1, update_gate, grad_next_h)
      grad_hidden_candidate:fill(1):addcmul(-1, hidden_candidate, hidden_candidate):cmul(grad_update_gate)
      local sub_Wh_t = Wh:narrow(2, 2 * outputsize + 1, outputsize):t()
      grad_update_gate:fill(0):addmm(grad_hidden_candidate, sub_Wh_t):cmul(prev_h)
      grad_reset_gate:fill(1):add(-1, reset_gate):cmul(reset_gate):cmul(grad_update_gate)

      --buffer:resizeAs(prev_h):copy(prev_h):add(-1, hidden_candidate)
      buffer:add(prev_h, -1, hidden_candidate);
      grad_update_gate:fill(1):add(-1, update_gate)
      grad_update_gate:cmul(update_gate):cmul(buffer):cmul(grad_next_h)
      grad_cur_x:mm(grad_gates, Wx:t())
      grad_Wx:addmm(scale, cur_x:t(), grad_gates)
      local sub_grad_gates = grad_gates:narrow(2, 1, 2 * outputsize)
      grad_Wh:narrow(2, 1, 2 * outputsize):addmm(scale, prev_h:t(), sub_grad_gates)

      buffer:resize(outputsize):sum(grad_gates, 1)
      grad_b:add(scale, buffer)
      buffer:cmul(prev_h, reset_gate)
      grad_Wh:narrow(2, 2 * outputsize + 1, outputsize):addmm(scale, buffer:t(), grad_hidden_candidate)
      grad_prev_h:cmul(grad_next_h, update_gate)
      grad_prev_h:addmm(sub_grad_gates, Wh:narrow(2, 1, 2 * outputsize):t())
      buffer:mm(grad_hidden_candidate, sub_Wh_t):cmul(reset_gate)
      grad_prev_h:add(buffer)
   end

   return self.gradInput
end

function StepGRU:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function StepGRU:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function StepGRU:clearState()
   self.gates:set()
   self.output:set()
   self.gradInput[1]:set()
   self.gradInput[2]:set()
end

function StepGRU:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

function StepGRU:parameters()
   return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
end

function StepGRU:maskZero(v1)
   self.maskzero = true
   self.v2 = not v1
   return self
end

StepGRU.setZeroMask = nn.MaskZero.setZeroMask

function StepGRU:__tostring__()
   return self.__typename .. string.format("(%d -> %d)", self.inputsize, self.outputsize)
end