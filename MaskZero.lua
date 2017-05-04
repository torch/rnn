------------------------------------------------------------------------
--[[ MaskZero ]]--
-- Zeroes the elements of the state tensors
-- (output/gradOutput/input/gradInput) of the encapsulated module
-- for commensurate elements that are 1 in self.zeroMask.
-- By default only output/gradOutput are zeroMasked.
-- self.zeroMask is set with setZeroMask(zeroMask).
-- Only works in batch-mode.
-- Note that when input/gradInput are zeroMasked, it is in-place
------------------------------------------------------------------------
local MaskZero, parent = torch.class("nn.MaskZero", "nn.Decorator")

function MaskZero:__init(module, v1, maskinput, maskoutput)
   parent.__init(self, module)
   assert(torch.isTypeOf(module, 'nn.Module'))
   self.maskinput = maskinput -- defaults to false
   self.maskoutput = maskoutput == nil and true or maskoutput -- defaults to true
   self.v2 = not v1
end

function MaskZero:updateOutput(input)
   if self.v2 then
      assert(self.zeroMask ~= nil, "MaskZero expecting zeroMask tensor or false")
   else -- backwards compat
      self.zeroMask = nn.utils.getZeroMaskBatch(input, self.zeroMask)
   end

   if self.maskinput and self.zeroMask then
      nn.utils.recursiveZeroMask(input, self.zeroMask)
   end

   -- forward through decorated module
   local output = self.modules[1]:updateOutput(input)

   if self.maskoutput and self.zeroMask then
      self.output = nn.utils.recursiveCopy(self.output, output)
      nn.utils.recursiveZeroMask(self.output, self.zeroMask)
   else
      self.output = output
   end

   return self.output
end

function MaskZero:updateGradInput(input, gradOutput)
   assert(self.zeroMask ~= nil, "MaskZero expecting zeroMask tensor or false")

   if self.maskoutput and self.zeroMask then
      self.gradOutput = nn.utils.recursiveCopy(self.gradOutput, gradOutput)
      nn.utils.recursiveZeroMask(self.gradOutput, self.zeroMask)
      gradOutput = self.gradOutput
   end

   self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)

   if self.maskinput and self.zeroMask then
      nn.utils.recursiveZeroMask(self.gradInput, self.zeroMask)
   end

   return self.gradInput
end

function MaskZero:clearState()
   self.output = nil
   self.gradInput = nil
   self.zeroMask = nil
   return self
end

function MaskZero:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

function MaskZero:setZeroMask(zeroMask)
   if zeroMask == false then
      self.zeroMask = false
   else
      assert(torch.isByteTensor(zeroMask))
      assert(zeroMask:isContiguous())
      self.zeroMask = zeroMask
   end
end
