------------------------------------------------------------------------
--[[ Repeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly
-- presented with the same input for seqlen time steps.
-- The output is a table of seqlen outputs of the rnn.
------------------------------------------------------------------------
assert(not nn.Repeater, "update nnx package : luarocks install nnx")
local Repeater, parent = torch.class('nn.Repeater', 'nn.AbstractSequencer')

function Repeater:__init(module, seqlen)
   parent.__init(self)
   assert(torch.type(seqlen) == 'number', "expecting number value for arg 2")
   self.seqlen = seqlen
   self.module = (not torch.isTypeOf(module, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module

   self.module:maxBPTTstep(seqlen) -- hijack seqlen (max number of time-steps for backprop)

   self.modules[1] = self.module
   self.output = {}
end

function Repeater:updateOutput(input)
   self.module = self.module or self.rnn -- backwards compatibility

   self.module:forget()
   -- TODO make copy outputs optional
   for step=1,self.seqlen do
      self.output[step] = nn.utils.recursiveCopy(self.output[step], self.module:updateOutput(input))
   end
   return self.output
end

function Repeater:updateGradInput(input, gradOutput)
   assert(self.module.step - 1 == self.seqlen, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.seqlen, "gradOutput should have seqlen elements")

   -- back-propagate through time (BPTT)
   for step=self.seqlen,1,-1 do
      local gradInput = self.module:updateGradInput(input, gradOutput[step])
      if step == self.seqlen then
         self.gradInput = nn.utils.recursiveCopy(self.gradInput, gradInput)
      else
         nn.utils.recursiveAdd(self.gradInput, gradInput)
      end
   end

   return self.gradInput
end

function Repeater:accGradParameters(input, gradOutput, scale)
   assert(self.module.step - 1 == self.seqlen, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.seqlen, "gradOutput should have seqlen elements")

   -- back-propagate through time (BPTT)
   for step=self.seqlen,1,-1 do
      self.module:accGradParameters(input, gradOutput[step], scale)
   end

end

function Repeater:maxBPTTstep(seqlen)
   self.seqlen = seqlen
   self.module:maxBPTTstep(seqlen)
end

function Repeater:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.module.step - 1 == self.seqlen, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.seqlen, "gradOutput should have seqlen elements")

   -- back-propagate through time (BPTT)
   for step=self.seqlen,1,-1 do
      self.module:accUpdateGradParameters(input, gradOutput[step], lr)
   end
end

function Repeater:__tostring__()
   local tab = '  '
   local line = '\n'
   local str = torch.type(self) .. ' {' .. line
   str = str .. tab .. '[  input,    input,  ...,  input  ]'.. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. '[output(1),output(2),...,output('..self.seqlen..')]' .. line
   str = str .. '}'
   return str
end
