-- DEPRECATED use SeqLSTM directly instead
local SeqLSTMP, parent = torch.class('nn.SeqLSTMP', 'nn.SeqLSTM')

function SeqLSTMP:__init(inputsize, hiddensize, outputsize)
   assert(inputsize and hiddensize and outputsize, "Expecting input, hidden and output size")
   parent.__init(self, inputsize, hiddensize, outputsize)
end
