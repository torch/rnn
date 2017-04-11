local RecLSTM, parent = torch.class('nn.RecLSTM', 'nn.LSTM')

function RecLSTM:__init(inputsize, outputsize, seqlen)
   parent.__init(self, inputsize, outputsize, seqlen)
end

function RecLSTM:buildModel()
   return nn.StepLSTM(self.inputSize, self.outputSize)
end

function RecLSTM:maskZero()
   assert(torch.isTypeOf(self.modules[1], 'nn.StepLSTM'))
   for i,stepmodule in pairs(self.sharedClones) do
      stepmodule.maskzero = true
   end
   self.modules[1].maskzero = true
   return self
end

function RecLSTM:buildGate()
   error"Not Implemented"
end

function RecLSTM:buildInputGate()
   error"Not Implemented"
end

function RecLSTM:buildForgetGate()
   error"Not Implemented"
end

function RecLSTM:buildHidden()
   error"Not Implemented"
end

function RecLSTM:buildCell()
   error"Not Implemented"
end

function RecLSTM:buildOutputGate()
   error"Not Implemented"
end