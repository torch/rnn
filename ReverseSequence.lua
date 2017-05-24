local ReverseSequence, parent = torch.class("nn.ReverseSequence", "nn.Module")

function ReverseSequence:updateOutput(input)
   local seqlen
   if torch.isTensor(input) then
      seqlen = input:size(1)
      self.output = torch.isTensor(self.output) and self.output or input.new()
      self.output:resizeAs(input)

      self._range = self._range or torch.isCudaTensor(input) and torch.CudaLongTensor() or torch.LongTensor()
      if self._range:nElement() ~= seqlen then
         self._range:range(seqlen,1,-1)
      end
      self.output:index(input, 1, self._range)
   else
      seqlen = #input
      self.output = torch.type(self.output) == 'table' and self.output or {}
      assert(torch.type(input) == 'table', "Expecting table or tensor at arg 1")

      -- empty output table
      for k,v in ipairs(self.output) do
         self.output[k] = nil
      end

      -- reverse input
      local k = 1
      for i=seqlen,1,-1 do
         self.output[k] = input[i]
         k = k + 1
      end
   end

   return self.output
end

function ReverseSequence:updateGradInput(input, gradOutput)
   local seqlen
   if torch.isTensor(input) then
      seqlen = input:size(1)
      self.gradInput = torch.isTensor(self.gradInput) and self.gradInput or input.new()
      self.gradInput:resizeAs(input)

      self.gradInput:index(gradOutput, 1, self._range)
   else
      seqlen = #input
      self.gradInput = torch.type(self.gradInput) == 'table' and self.gradInput or {}
      assert(torch.type(gradOutput) == 'table', "Expecting table or tensor at arg 2")

      -- empty gradInput table
      for k,v in ipairs(self.gradInput) do
         self.gradInput[k] = nil
      end

      -- reverse gradOutput
      local k = 1
      for i=seqlen,1,-1 do
         self.gradInput[k] = gradOutput[i]
         k = k + 1
      end
   end

   return self.gradInput
end

function ReverseSequence:clearState()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self._range = nil
end

function ReverseSequence:type(...)
   self:clearState()
   return parent.type(self, ...)
end

