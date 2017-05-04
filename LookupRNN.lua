local LookupRNN, parent = torch.class("nn.LookupRNN", "nn.Recurrence")

function LookupRNN:__init(nindex, outputsize, transfer, merge)
   transfer = transfer or nn.Sigmoid()
   merge = merge or nn.CAddTable()
   local stepmodule = nn.Sequential() -- input is {x[t], h[t-1]}
      :add(nn.ParallelTable()
	      :add(nn.LookupTableMaskZero(nindex, outputsize)) -- input layer
         :add(nn.Linear(outputsize, outputsize))) -- recurrent layer
	  :add(merge)
	  :add(transfer)
   parent.__init(self, stepmodule, outputsize, 0)
   self.nindex = nindex
   self.outputsize = outputsize
end

function LookupRNN:__tostring__()
   return torch.type(self) .. "(" .. self.nindex .. " -> " .. self.outputsize ..")"
end