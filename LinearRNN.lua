
local LinearRNN, parent = torch.class("nn.LinearRNN", "nn.Recurrence")

function LinearRNN:__init(inputsize, outputsize, transfer)
   transfer = transfer or nn.Sigmoid()
   local stepmodule = nn.Sequential()
      :add(nn.JoinTable(1,1))
      :add(nn.Linear(inputsize+outputsize, outputsize))
      :add(transfer)
   parent.__init(self, stepmodule, outputsize, 1)
   self.inputsize = inputsize
   self.outputsize = outputsize
end

function LinearRNN:__tostring__()
   return torch.type(self) .. "(" .. self.inputsize .. " -> " .. self.outputsize ..")"
end