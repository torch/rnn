local _ = require 'moses'
local AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')

AbstractRecurrent.dpnn_stepclone = true

function AbstractRecurrent:__init(stepmodule)
   parent.__init(self)

   assert(torch.isTypeOf(stepmodule, 'nn.Module'), torch.type(self).." expecting nn.Module instance at arg 1")
   self.seqlen = 99999 --the maximum number of time steps to BPTT

   self.outputs = {}
   self.gradInputs = {}
   self._gradOutputs = {}
   self.gradOutputs = {}

   self.step = 1

   -- stores internal states of Modules at different time-steps
   self.modules[1] = stepmodule
   self.sharedClones = {stepmodule}
end

function AbstractRecurrent:getStepModule(step)
   step = step or 1
   local stepmodule = self.sharedClones[step]
   if not stepmodule then
      stepmodule = self.modules[1]:stepClone()
      self.sharedClones[step] = stepmodule
      self.nSharedClone = _.size(self.sharedClones)
   end
   return stepmodule
end

function AbstractRecurrent:updateOutput(input)
   if self.zeroMask then
      -- where zeroMask = 1, the past is forgotten, that is, the output/gradOutput is zeroed
      local stepmodule = (self.train==false) and self.modules[1] or self:getStepModule(self.step)
      self.zeroMaskStep = self.zeroMaskStep + 1
      if self.zeroMaskStep > self.zeroMask:size(1) then
         error"AbstractRecurrent.updateOutput called more times than self.zeroMask:size(1)"
      end
      stepmodule:setZeroMask(self.zeroMask[self.zeroMaskStep])
   end

   -- feed-forward for one time-step
   self.output = self:_updateOutput(input)

   self.outputs[self.step] = self.output

   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil

   return self.output
end

function AbstractRecurrent:updateGradInput(input, gradOutput)
   if self.zeroMask and self.zeroMask:size(1) ~= self.zeroMaskStep then
      error"AbstractRecurrent.updateOutput called less times than self.zeroMask:size(1)"
   end

   -- updateGradInput should be called in reverse order of time
   self.updateGradInputStep = self.updateGradInputStep or self.step

   -- BPTT for one time-step
   self.gradInput = self:_updateGradInput(input, gradOutput)

   self.updateGradInputStep = self.updateGradInputStep - 1
   self.gradInputs[self.updateGradInputStep] = self.gradInput
   return self.gradInput
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   -- accGradParameters should be called in reverse order of time
   assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
   self.accGradParametersStep = self.accGradParametersStep or self.step

   -- BPTT for one time-step
   self:_accGradParameters(input, gradOutput, scale)

   self.accGradParametersStep = self.accGradParametersStep - 1
end

-- goes hand in hand with forget()
-- this methods brings the oldest memory to the current step
function AbstractRecurrent:recycle()
   self.nSharedClone = self.nSharedClone or _.size(self.sharedClones)

   local seqlen = math.max(self.seqlen + 1, self.nSharedClone)
   if self.sharedClones[self.step] == nil then
      self.sharedClones[self.step] = self.sharedClones[self.step-seqlen]
      self.sharedClones[self.step-seqlen] = nil
      self._gradOutputs[self.step] = self._gradOutputs[self.step-seqlen]
      self._gradOutputs[self.step-seqlen] = nil
   end

   self.outputs[self.step-seqlen-1] = nil
   self.gradInputs[self.step-seqlen-1] = nil

   return self
end

function nn.AbstractRecurrent:clearState()
   self:forget()
   -- keep the first two sharedClones
   nn.utils.clear(self, 'prevOutput', '_input', '_gradOutput', '_gradOutputs', 'gradPrevOutput', 'cell', 'cells', 'gradCells', 'outputs', 'gradInputs', 'gradOutputs')
   for i, clone in ipairs(self.sharedClones) do
      clone:clearState()
   end
   self.modules[1]:clearState()
   self.zeroMask = nil
   return parent.clearState(self)
end

-- sets the starting hidden state at time t=0 (that is h[0])
function AbstractRecurrent:setStartState(startState)
   self.startState = startState
end

-- this method brings all the memory back to the start
function AbstractRecurrent:forget()
   -- the stepmodule may contain an AbstractRecurrent instance (issue 107)
   parent.forget(self)

    -- bring all states back to the start of the sequence buffers
   if self.train ~= false then
      self.outputs = {}
      self.gradInputs = {}
      self.sharedClones = _.compact(self.sharedClones)
      self._gradOutputs = _.compact(self._gradOutputs)
      self.gradOutputs = {}
      if self.cells then
         self.cells = {}
         self.gradCells = {}
      end
   end

   -- forget the past inputs; restart from first step
   self.step = 1

   if not self.rmInSharedClones then
      -- Asserts that issue 129 is solved. In forget as it is often called.
      -- Asserts that self.modules[1] is part of the sharedClones.
      -- Since its used for evaluation, it should be used for training.
      local nClone, maxIdx = 0, 1
      for k,v in pairs(self.sharedClones) do -- to prevent odd bugs
         if torch.pointer(v) == torch.pointer(self.modules[1]) then
            self.rmInSharedClones = true
            maxIdx = math.max(k, maxIdx)
         end
         nClone = nClone + 1
      end
      if nClone > 1 then
         if not self.rmInSharedClones then
            print"WARNING : modules[1] should be added to sharedClones in constructor."
            print"Adding it for you."
            assert(torch.type(self.sharedClones[maxIdx]) == torch.type(self.modules[1]))
            self.modules[1] = self.sharedClones[maxIdx]
            self.rmInSharedClones = true
         end
      end
   end
   return self
end

function AbstractRecurrent:includingSharedClones(f)
   local modules = self.modules
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   self.modules = {}
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules or {}) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end

function AbstractRecurrent:type(type, tensorcache)
   return self:includingSharedClones(function()
      return parent.type(self, type, tensorcache)
   end)
end

function AbstractRecurrent:maskZero(v1)
   if not self.maskzero then
      self.maskzero = true
      local stepmodule = nn.MaskZero(self.modules[1], v1)
      self.sharedClones = {stepmodule}
      self.modules[1] = stepmodule
   end
   return self
end

function AbstractRecurrent:setZeroMask(zeroMask)
   if zeroMask == false then
      self.zeroMask = false
      for k,stepmodule in pairs(self.sharedClones) do
         stepmodule:setZeroMask(zeroMask)
      end
   elseif torch.isTypeOf(self.modules[1], 'nn.AbstractRecurrent') then
      self.modules[1]:setZeroMask(zeroMask)
   else
      assert(zeroMask:dim() >= 2, "Expecting dim >= 2 for zeroMask. For example, seqlen x batchsize")
      -- reserve for later. Each step will be masked in updateOutput
      self.zeroMask = zeroMask
      self.zeroMaskStep = 0
   end
end

function AbstractRecurrent:training()
   return self:includingSharedClones(function()
      return parent.training(self)
   end)
end

function AbstractRecurrent:evaluate()
   return self:includingSharedClones(function()
      return parent.evaluate(self)
   end)
end

function AbstractRecurrent:reinforce(reward)
   local a = torch.Timer()
   if torch.type(reward) == 'table' then
      -- multiple rewards, one per time-step
      local rstep = #reward
      assert(self.step > rstep)
      for step=self.step-1,self.step-#reward,-1 do
         local sm = self:getStepModule(step)
         sm:reinforce(reward[rstep])
         rstep = rstep - 1
      end
   elseif torch.isTensor(reward) and reward:dim() >= 2 then
      -- multiple rewards, one per time-step
      local rstep = reward:size(1)
      assert(self.step > rstep)
      for step=self.step-1,self.step-reward:size(1),-1 do
         local sm = self:getStepModule(step)
         sm:reinforce(reward[rstep])
         rstep = rstep - 1
      end
   else
      -- one reward broadcast to all time-steps
      return self:includingSharedClones(function()
         return parent.reinforce(self, reward)
      end)
   end
end

-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function AbstractRecurrent:setOutputStep(step)
   self.output = self.outputs[step] --or self:getStepModule(step).output
   assert(self.output, "no output for step "..step)
   self.gradInput = self.gradInputs[step]
end

function AbstractRecurrent:maxBPTTstep(seqlen)
   self.seqlen = seqlen
end

-- get stored hidden state: h[t] where h[t] = f(x[t], h[t-1])
function AbstractRecurrent:getHiddenState(step, input)
   error"Not Implemented"
end

-- set stored hidden state
function AbstractRecurrent:setHiddenState(step, hiddenState)
   error"Not Implemented"
end

-- get stored grad hidden state: grad(h[t])
function AbstractRecurrent:getGradHiddenState(step, input)
   error"Not Implemented"
end

-- set stored grad hidden state
function AbstractRecurrent:setGradHiddenState(step, gradHiddenState)
   error"Not Implemented"
end

function AbstractRecurrent:__tostring__()
   local inputsize = self.inputsize or self.inputSize
   local outputsize = self.outputsize or self.outputSize
   if inputsize and outputsize then
       return self.__typename .. string.format("(%d -> %d)", inputsize, outputsize)
   else
       return self.__typename
   end
end
