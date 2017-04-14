-- StepLSTM is a step-wise module that can be used inside Recurrence to implement an LSTM.
-- That is, the StepLSTM efficiently implements a single LSTM time-step.
-- Its efficient because it doesn't use any internal modules; it calls BLAS directly.
-- StepLSTM is based on SeqLSTM.
-- Input : {input[t], hidden[t-1], cell[t-1])}
-- Output: {hidden[t], cell[t]}
local StepLSTM, parent = torch.class('nn.StepLSTM', 'nn.Module')

function StepLSTM:__init(inputsize, outputsize)
   parent.__init(self)
   self.inputsize, self.outputsize = inputsize, outputsize

   self.weight = torch.Tensor(inputsize+outputsize, 4 * outputsize)
   self.gradWeight = torch.Tensor(inputsize+outputsize, 4 * outputsize)

   self.bias = torch.Tensor(4 * outputsize)
   self.gradBias = torch.Tensor(4 * outputsize):zero()
   self:reset()

   self.gates = torch.Tensor() -- batchsize x 4*outputsize

   self.output = {torch.Tensor(), torch.Tensor()}
   self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
end


function StepLSTM:reset(std)
   if not std then
      std = 1.0 / math.sqrt(self.outputsize + self.inputsize)
   end
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   self.weight:normal(0, std)
   return self
end

-- unlike MaskZero, the mask is applied in-place
function StepLSTM:recursiveMask(output, mask)
   if torch.type(output) == 'table' then
      for k,v in ipairs(output) do
         self:recursiveMask(output[k], mask)
      end
   else
      assert(torch.isTensor(output))

      -- make sure mask has the same dimension as the output tensor
      local outputSize = output:size():fill(1)
      outputSize[1] = output:size(1)
      mask:resize(outputSize)
      -- build mask
      local zeroMask = mask:expandAs(output)
      output:maskedFill(zeroMask, 0)
   end
end


function StepLSTM:updateOutput(input)
   self.recompute_backward = true
   local cur_x, prev_h, prev_c = input[1], input[2], input[3]
   assert(torch.isTensor(prev_h))
   assert(torch.isTensor(prev_c))
   local batchsize, inputsize, outputsize = cur_x:size(1), cur_x:size(2), self.outputsize
   assert(inputsize == self.inputsize)

   local bias_expand = self.bias:view(1, 4 * outputsize):expand(batchsize, 4 * outputsize)
   local Wx = self.weight:narrow(1,1,inputsize)
   local Wh = self.weight:narrow(1,inputsize+1,outputsize)

   local next_h, next_c = self.output[1], self.output[2]
   next_h:resize(batchsize, outputsize)
   next_c:resize(batchsize, outputsize)

   self.gates:resize(batchsize, 4 * outputsize):zero()
   local gates = self.gates

   -- forward
   gates:addmm(bias_expand, cur_x, Wx)
   gates:addmm(prev_h, Wh)
   gates[{{}, {1, 3 * outputsize}}]:sigmoid()
   gates[{{}, {3 * outputsize + 1, 4 * outputsize}}]:tanh()
   local input_gate = gates[{{}, {1, outputsize}}]
   local forget_gate = gates[{{}, {outputsize + 1, 2 * outputsize}}]
   local output_gate = gates[{{}, {2 * outputsize + 1, 3 * outputsize}}]
   local input_transform = gates[{{}, {3 * outputsize + 1, 4 * outputsize}}]
   next_h:cmul(input_gate, input_transform)
   next_c:cmul(forget_gate, prev_c):add(next_h)
   next_h:tanh(next_c):cmul(output_gate)

   if self.maskzero then
      -- build mask from input
      local zero_mask = torch.getBuffer('StepLSTM', 'zero_mask', cur_x)
      zero_mask:norm(cur_x, 2, 2)
      self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaByteTensor() or torch.ByteTensor())
      zero_mask.eq(self.zeroMask, zero_mask, 0)
      -- zero masked output
      self:recursiveMask({next_h, next_c, cur_gates}, self.zeroMask)
   end

   return self.output
end

function StepLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   local cur_x, prev_h, prev_c = input[1], input[2], input[3]
   local batchsize, inputsize, outputsize = cur_x:size(1), cur_x:size(2), self.outputsize
   assert(inputsize == self.inputsize)

   local grad_next_h, grad_next_c = gradOutput[1], gradOutput[2]
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   local grad_cur_x, grad_prev_h, grad_prev_c = self.gradInput[1], self.gradInput[2], self.gradInput[3]
   grad_cur_x:resize(batchsize, inputsize)
   grad_prev_h:resize(batchsize, outputsize)
   grad_prev_c:resize(batchsize, outputsize)

   local next_h, next_c = self.output[1], self.output[2]

   local Wx = self.weight:narrow(1,1,inputsize)
   local Wh = self.weight:narrow(1,inputsize+1,outputsize)
   local grad_Wx = self.gradWeight:narrow(1,1,inputsize)
   local grad_Wh = self.gradWeight:narrow(1,inputsize+1,outputsize)
   local grad_b = self.gradBias

   local gates = self.gates

   local grad_gates = torch.getBuffer('StepLSTM', 'grad_gates', gates) -- batchsize x 4*outputsize
   local grad_gates_sum = torch.getBuffer('StepLSTM', 'grad_gates_sum', gates) -- 1 x 4*outputsize

   -- backward
   if self.maskzero then
      -- zero masked gradOutput
      self:recursiveMask({grad_next_h, grad_next_c}, self.zeroMask)
   end

   local input_gate = gates[{{}, {1, outputsize}}]
   local forget_gate = gates[{{}, {outputsize + 1, 2 * outputsize}}]
   local output_gate = gates[{{}, {2 * outputsize + 1, 3 * outputsize}}]
   local input_transform = gates[{{}, {3 * outputsize + 1, 4 * outputsize}}]

   grad_gates:resize(batchsize, 4 * outputsize):zero()
   local grad_input_gate = grad_gates[{{}, {1, outputsize}}]
   local grad_forget_gate = grad_gates[{{}, {outputsize + 1, 2 * outputsize}}]
   local grad_output_gate = grad_gates[{{}, {2 * outputsize + 1, 3 * outputsize}}]
   local grad_input_transform = grad_gates[{{}, {3 * outputsize + 1, 4 * outputsize}}]

   -- we use grad_[input,forget,output]_gate as temporary buffers to compute grad_prev_c.
   grad_input_gate:tanh(next_c)
   grad_forget_gate:cmul(grad_input_gate, grad_input_gate)
   grad_output_gate:fill(1):add(-1, grad_forget_gate):cmul(output_gate):cmul(grad_next_h)
   grad_prev_c:add(grad_next_c, grad_output_gate)

   -- we use above grad_input_gate to compute grad_output_gate
   grad_output_gate:fill(1):add(-1, output_gate):cmul(output_gate):cmul(grad_input_gate):cmul(grad_next_h)

   -- Use grad_input_gate as a temporary buffer for computing grad_input_transform
   grad_input_gate:cmul(input_transform, input_transform)
   grad_input_transform:fill(1):add(-1, grad_input_gate):cmul(input_gate):cmul(grad_prev_c)

   -- We don't need any temporary storage for these so do them last
   grad_input_gate:fill(1):add(-1, input_gate):cmul(input_gate):cmul(input_transform):cmul(grad_prev_c)
   grad_forget_gate:fill(1):add(-1, forget_gate):cmul(forget_gate):cmul(prev_c):cmul(grad_prev_c)

   grad_cur_x:mm(grad_gates, Wx:t())
   grad_Wx:addmm(scale, cur_x:t(), grad_gates)
   grad_Wh:addmm(scale, prev_h:t(), grad_gates)
   grad_gates_sum:resize(1, 4 * outputsize):sum(grad_gates, 1)
   grad_b:add(scale, grad_gates_sum)

   grad_prev_h:mm(grad_gates, Wh:t())
   grad_prev_c:cmul(forget_gate)

   return self.gradInput
end

function StepLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function StepLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end


function StepLSTM:clearState()
   self.gates:set()

   self.output[1]:set(); self.output[2]:set()
   self.gradInput[1]:set(); self.gradInput[2]:set(); self.gradInput[3]:set()

   self.zeroMask = nil
   self._zeroMask = nil
   self._maskbyte = nil
   self._maskindices = nil
end

function StepLSTM:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

StepLSTM.toFastLSTM = nn.SeqLSTM.toFastLSTM