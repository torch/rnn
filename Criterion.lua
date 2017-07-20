local Criterion = nn.Criterion

Criterion.toBatch = nn.Module.toBatch
Criterion.fromBatch = nn.Module.fromBatch


function Criterion:setZeroMask(zeroMask)
   if self.criterions then
      for i, criterion in ipairs(self.criterions) do
         criterion:setZeroMask(zeroMask)
      end
   end
   if self.criterion then
   	  self.criterion:setZeroMask(zeroMask)
   end
end

function Criterion:clearState()
   return nn.utils.clear(self, 'gradInput')
end