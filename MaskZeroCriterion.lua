------------------------------------------------------------------------
--[[ MaskZeroCriterion ]]--
-- Decorator that zeros err and gradInputs of the encapsulated criterion
-- for commensurate input rows which are tensors of zeros
------------------------------------------------------------------------
local MaskZeroCriterion, parent = torch.class("nn.MaskZeroCriterion", "nn.Criterion")

function MaskZeroCriterion:__init(criterion, v1)
   parent.__init(self)
   self.criterion = assert(criterion)
   assert(torch.isTypeOf(criterion, 'nn.Criterion'))
   self.v2 = not v1
end

function MaskZeroCriterion:updateOutput(input, target)
   if self.v2 then
      assert(self.zeroMask ~= nil, "MaskZeroCriterion expecting zeroMask tensor or false")
      if self.zeroMask == false then
         self.output = self.criterion:updateOutput(input, target)
         return self.output
      end
      assert(self.zeroMask:dim() == 1, "MaskZeroCriterion expecting zeroMask of size batchsize")
   else -- backwards compat
      self.zeroMask = nn.utils.getZeroMaskBatch(input, self.zeroMask)
   end

   self.isEmptyBatch = (self.zeroMask:sum() == self.zeroMask:nElement())
   if self.isEmptyBatch then
      self.output = 0
   else
      local first = nn.utils.recursiveGetFirst(input)
      -- e.g. 0,1,0 -> 1,0,1
      self._oneMask = self._oneMask or self.zeroMask.new()
      self._oneMask:lt(self.zeroMask, 1)
      -- 1,0,1 -> 1,3
      self._indices = self._indices or torch.isCudaTensor(first) and torch.CudaLongTensor() or torch.LongTensor()
      self._range = self._range or self._indices.new()
      self._range:range(1,self._oneMask:nElement())
      self._indices:maskedSelect(self._range, self._oneMask)
      -- indexSelect the input
      self.input = nn.utils.recursiveIndex(self.input, input, 1, self._indices)
      self.target = nn.utils.recursiveIndex(self.target, target, 1, self._indices)
      -- forward through decorated criterion
      self.output = self.criterion:updateOutput(self.input, self.target)
   end

   return self.output
end

function MaskZeroCriterion:updateGradInput(input, target)
   if self.zeroMask == false then
      self.gradInput = self.criterion:updateGradInput(input, target)
      return self.gradInput
   end

   self._gradInput = nn.utils.recursiveResizeAs(self._gradInput, input)
   nn.utils.recursiveFill(self._gradInput, 0)

   if not self.isEmptyBatch then
      assert(self.input and self.target)
      local gradInput = self.criterion:updateGradInput(self.input, self.target)
      nn.utils.recursiveIndexCopy(self._gradInput, 1, self._indices, gradInput)
   end

   self.gradInput = self._gradInput
   return self.gradInput
end

function MaskZeroCriterion:clearState()
   self.zeroMask = nil
   self._oneMask = nil
   self._range = nil
   self._indices = nil
   self.input = nil
   self.target = nil
   self.output = nil
   self.gradInput = nil
   self._gradInput = nil
   self.criterion:clearState()
   return parent.clearState(self)
end

function MaskZeroCriterion:type(type, ...)
   self:clearState()
   self.criterion:type(type, ...)
   return parent.type(self, type, ...)
end

function MaskZeroCriterion:setZeroMask(zeroMask)
   self.zeroMask = zeroMask
end
