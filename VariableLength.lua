local VariableLength, parent = torch.class("nn.VariableLength", "nn.Decorator")

-- This modules deals with variable lengths representations
-- of input data. It takes the simplest representation of variable length sequences:
--
-- {
--    torch.Tensor(T1, [...]),
--    torch.Tensor(T2, [...]),
--    ...,
--    torch.Tensor(TN, [...])
-- }
--
-- and turns it either into an equivalent (in terms of amount of info)
-- "sparse/size" representation:
--
-- "sparse": torch.Tensor(T1+...+TN, [...])
-- "size": torch.LongTensor({T1,T2,...,TN})
--
-- where "sparse" is the direct concatenation of the input array of tensors
-- and "size" is the 1D tensor containing the sequence lengths
--
-- or  into an equivalent (in terms of amount of info)
-- "dense/masked" representation:
--
-- "dense": torch.Tensor(max_i{T}, N, [...])
-- "mask": torch.ByteTensor(max_i{T}, N)
-- where max_i{T} is the maximum sequence length,
-- "dense" is the rectangular version of the input data,
-- "mask" indicates where a sequence ends (1) or is valid (0)
function VariableLength:__init(module, lastOnly, sparse)
   parent.__init(self, module)
   -- only extract the last element of each sequence
   self.lastOnly = lastOnly -- defaults to false
   if sparse or torch.type(module) == 'nn.VSeqLSTM' then
      self.sparse = true
      self.sizes = torch.LongTensor()
      self.output = torch.Tensor()
      self.mapping = torch.LongTensor()
      self.sorted_by_batch_sizes = torch.LongTensor()
      self.sorted_by_time_sizes = torch.LongTensor()
      self.sorted_by_time_indices = torch.LongTensor()
      self.cinput_bis = torch.Tensor()
   end
   self.gradInput = {}
end

function VariableLength:updateOutput(input)
   -- input is a table of batchSize tensors
   assert(torch.type(input) == 'table')
   assert(torch.isTensor(input[1]))
   local batchSize = #input

   if self.sparse then
      -- Path for "sparse/size" representation of arrays of tensors,
      -- where an array of tensors is transformed into its
      -- sparse/size equivalent representation, i.e.:
      -- {
      --    torch.Tensor(T1, ...),
      --    torch.Tensor(T2, ...),
      --    ...,
      --    torch.Tensor(TN, ...)
      -- }
      -- -->
      -- { torch.Tensor(T1+...+TN, ...), torch.LongTensor({T1,T2,...,TN) }


      -- Initialize a bunch of buffers
      local first_input = input[1]
      self.cinput = self.cinput or first_input.new()
      self.cinput = self.cinput:type(first_input:type())
      self.cgradInput = self.cgradInput or first_input.new()
      self.cgradInput = self.cgradInput:type(first_input:type())
      self._input = self._input or first_input.new()
      self._input = self._input:type(first_input:type())
      self._input = self._input or {}

      -- Concatenate the array of tensors,
      -- extract the sequence sizes
      local sm = 0
      local mx = 0
      self.cinput:cat(input, 1)
      self.sizes:resize(#input)
      for i=1,#input do
         self.sizes[i] = input[i]:size(1)
      end

      -- From the concatenated, batch-first 'self.cinput', along with 'self.sizes',
      -- transpose to a time-first, sorted in decreasing order of batch size
      -- to 'self._input'
      self.cinput.THRNN.LSTM_bt_to_sorted_tb(
                self.cinput:cdata(),
                self.sizes:cdata(),
                self._input:cdata(),
                self.mapping:cdata(),
                self.sorted_by_batch_sizes:cdata(),
                self.sorted_by_time_sizes:cdata(),
                self.sorted_by_time_indices:cdata(),
                0)

      -- Set the context for all the modules,
      -- containing the sorted sizes
      self.context = self.context or {}
      self.context.sizes = self.sorted_by_batch_sizes
      self.modules[1]:setContext(self.context)

      -- Run the wrapped module
      local output = self.modules[1]:updateOutput(self._input)

      if self.lastOnly then
         -- Extract the last time step of each sample.
         -- self.output tensor has shape: batchSize [x outputSize]
         self.output = torch.isTensor(self.output) and self.output or output.new()
         self.cinput.THRNN.LSTM_sorted_tb_to_bt(
                   output:cdata(),
                   self.sizes:cdata(),
                   self.mapping:cdata(),
                   self.output:cdata(),
                   1)
      else
         -- Reverse the transpose
         self.output = {}
         self._output = self._output or first_input.new()
         self._output = self._output:type(first_input:type())
         output.THRNN.LSTM_sorted_tb_to_bt(
                   output:cdata(),
                   self.sizes:cdata(),
                   self.mapping:cdata(),
                   self._output:cdata(),
                   0)
         local runningIdx = 1
         for i=1,#input do
            self.output[i] = self._output:narrow(1, runningIdx, self.sizes[i])
            runningIdx = runningIdx + self.sizes[i]
         end
      end

   else
      -- Path for "dense/masked" representations of arrays of tensors,
      -- where an array of tensors is transformed into its
      -- dense/masked equivalent representation, i.e.:
      -- {
      --    torch.Tensor(T1, ...),
      --    torch.Tensor(T2, ...),
      --    ...,
      --    torch.Tensor(TN, ...)
      -- }
      -- -->
      -- { torch.Tensor(max_i{T}, N, ...), torch.ByteTensor(max_i{T}, N) }
      -- where max_i{T} is the maximum sequence length

      self._input = self._input or input[1].new()
      -- mask is a binary tensor with 1 where self._input is zero (between sequence zero-mask)
      self._mask = self._mask or torch.ByteTensor()

      -- now we process input into _input.
      -- indexes and mappedLengths are meta-information tables, explained below.
      self.indexes, self.mappedLengths = self._input.nn.VariableLength_FromSamples(input, self._input, self._mask)

      -- zero-mask the _input where mask is 1
      nn.utils.recursiveZeroMask(self._input, self._mask)
      self.modules[1]:setZeroMask(self._mask)

      -- feedforward the zero-mask format through the decorated module
      local output = self.modules[1]:updateOutput(self._input)

      if self.lastOnly then
         -- Extract the last time step of each sample.
         -- self.output tensor has shape: batchSize [x outputSize]
         self.output = torch.isTensor(self.output) and self.output or output.new()
         self.output.nn.VariableLength_ToFinal(self.indexes, self.mappedLengths, output, self.output)
      else
         -- This is the revese operation of everything before updateOutput
         self.output = self._input.nn.VariableLength_ToSamples(self.indexes, self.mappedLengths, output)
      end
   end
   return self.output
end

function VariableLength:updateGradInput(input, gradOutput)

   assert(torch.type(input) == 'table')
   assert(torch.isTensor(input[1]))
   if self.sparse then
      -- Path for "sparse/size" representation of arrays of tensors,
      -- where an array of tensors is transformed into its
      -- sparse/size equivalent representation, i.e.:
      -- {
      --    torch.Tensor(T1, ...),
      --    torch.Tensor(T2, ...),
      --    ...,
      --    torch.Tensor(TN, ...)
      -- }
      -- -->
      -- { torch.Tensor(T1+...+TN, ...), torch.LongTensor({T1,T2,...,TN) }
      local first_input = input[1]
      self._gradOutput = self._gradOutput or first_input.new()
      self.cinput = self.cinput or first_input.new()
      self.cinput = self.cinput:type(first_input:type())

      if self.lastOnly then
         -- Call the transposer with the "last" argument == 1
         self.cinput.THRNN.LSTM_bt_to_sorted_tb(
                   gradOutput:cdata(),
                   self.sizes:cdata(),
                   self._gradOutput:cdata(),
                   self.mapping:cdata(),
                   self.sorted_by_batch_sizes:cdata(),
                   self.sorted_by_time_sizes:cdata(),
                   self.sorted_by_time_indices:cdata(),
                   1)
      else
         -- Concatenate the gradOutput,
         -- and call the transposer
         self.cinput:cat(gradOutput, 1)
         self.cinput.THRNN.LSTM_bt_to_sorted_tb(
                   self.cinput:cdata(),
                   self.sizes:cdata(),
                   self._gradOutput:cdata(),
                   self.mapping:cdata(),
                   self.sorted_by_batch_sizes:cdata(),
                   self.sorted_by_time_sizes:cdata(),
                   self.sorted_by_time_indices:cdata(),
                   0)
      end
      -- updateGradInput decorated module
      self.context = self.context or {}
      self.context.sizes = self.sorted_by_batch_sizes
      self.modules[1]:setContext(self.context)
      local gradInput = self.modules[1]:updateGradInput(self._input, self._gradOutput)

      -- Final call to the de-transposer before returning
      self.gradInput = {}
      self._gradInput = self._gradInput or first_input.new()
      self._gradInput = self._gradInput:type(first_input:type())
      self.cinput.THRNN.LSTM_sorted_tb_to_bt(
                gradInput:cdata(),
                self.sizes:cdata(),
                self.mapping:cdata(),
                self._gradInput:cdata(),
                0)

      local runningIdx = 1
      for i=1,#input do
         self.gradInput[i] = self._gradInput:narrow(1,runningIdx, self.sizes[i])
         runningIdx = runningIdx + self.sizes[i]
      end
   else
      -- Path for "dense/masked" representations of arrays of tensors,
      -- where an array of tensors is transformed into its
      -- dense/masked equivalent representation, i.e.:
      -- {
      --    torch.Tensor(T1, ...),
      --    torch.Tensor(T2, ...),
      --    ...,
      --    torch.Tensor(TN, ...)
      -- }
      -- -->
      -- { torch.Tensor(max_i{T}, N, ...), torch.ByteTensor(max_i{T}, N) }
      -- where max_i{T} is the maximum sequence length
      self._gradOutput = self._gradOutput or self._input.new()
      if self.lastOnly then
         assert(torch.isTensor(gradOutput))
         self._gradOutput.nn.VariableLength_FromFinal(self.indexes, self.mappedLengths, gradOutput, self._gradOutput)
      else
         assert(torch.type(gradOutput) == 'table')
         assert(torch.isTensor(gradOutput[1]))
         self.indexes, self.mappedLengths = self._gradOutput.nn.VariableLength_FromSamples(gradOutput, self._gradOutput, self._mask)
      end

      -- zero-mask the _gradOutput where mask is 1
      nn.utils.recursiveZeroMask(self._gradOutput, self._mask)

      -- updateGradInput decorated module
      local gradInput = self.modules[1]:updateGradInput(self._input, self._gradOutput)

      self.gradInput = self._input.nn.VariableLength_ToSamples(self.indexes, self.mappedLengths, gradInput)
   end

   return self.gradInput
end

function VariableLength:accGradParameters(input, gradOutput, scale)
   -- requires a previous call to updateGradInput
   self.modules[1]:accGradParameters(self._input, self._gradOutput, scale)
end

function VariableLength:clearState()
   self.gradInput = {}
   if torch.isTensor(self.output) then
      self.output:set()
   else
      self.output = {}
   end
   self._gradOutput = nil
   self._input = nil
   return parent.clearState(self)
end

function VariableLength:setZeroMask()
   error"Not Supported"
end
