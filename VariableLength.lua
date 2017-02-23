local VariableLength, parent = torch.class("nn.VariableLength", "nn.Decorator")

-- make sure your module has been set-up for zero-masking (that is, module:maskZero())
function VariableLength:__init(module, lastOnly)
   parent.__init(self, module)
   -- only extract the last element of each sequence
   self.lastOnly = lastOnly -- defaults to false
end

-- recursively masks input (inplace)
function VariableLength.recursiveMask(input, mask)
   if torch.type(input) == 'table' then
      for k,v in ipairs(input) do
         self.recursiveMask(v, mask)
      end
   else
      assert(torch.isTensor(input))

      -- make sure mask has the same dimension as the input tensor
      assert(mask:dim() == 2, "Expecting batchsize x seqlen mask tensor")
      -- expand mask to input (if necessary)
      local zeroMask
      if input:dim() == 2 then
         zeroMask = mask
      elseif input:dim() > 2 then
         local inputSize = input:size():fill(1)
         inputSize[1] = input:size(1)
         inputSize[2] = input:size(2)
         mask:resize(inputSize)
         zeroMask = mask:expandAs(input)
      else
         error"Expecting batchsize x seqlen [ x ...] input tensor"
      end
      -- zero-mask input in between sequences
      input:maskedFill(zeroMask, 0)
   end
end

function VariableLength:updateOutput(input)
   -- input is a table of batchSize tensors
   assert(torch.type(input) == 'table')
   assert(torch.isTensor(input[1]))
   local batchSize = #input

   self._input = self._input or input[1].new()
   -- mask is a binary tensor with 1 where self._input is zero (between sequence zero-mask)
   self._mask = self._mask or torch.ByteTensor()

   -- now we process input into _input.
   -- indexes and mappedLengths are meta-information tables, explained below.
   self.indexes, self.mappedLengths = self._input.nn.VariableLength_FromSamples(input, self._input, self._mask)

   -- zero-mask the _input where mask is 1
   self.recursiveMask(self._input, self._mask)

   -- feedforward the zero-mask format through the decorated module
   local output = self.modules[1]:updateOutput(self._input)

   if self.lastOnly then
      -- Extract the last time step of each sample.
      -- self.output tensor has shape: batchSize [x outputSize]
      self.output = torch.isTensor(self.output) and self.output or output.new()
      self.output.nn.VariableLength_ToFinal(selfindexes, self.mappedLengths, output, self.output)
   else
      -- This is the revese operation of everything before updateOutput
      self.output = input.nn.VariableLength_ToSamples(self.indexes, self.mappedLengths, output)
   end

   return self.output
end

function VariableLength:updateGradInput(input, gradInput)

   return self.gradInput
end

function VariableLength:accGradParameters(input, gradInput, scale)

end