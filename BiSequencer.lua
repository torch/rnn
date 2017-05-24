------------------------------------------------------------------------
--[[ BiSequencer ]]--
-- Encapsulates forward, backward and merge modules.
-- Input is a seqlen x inputsize [x ...] sequence tensor
-- Output is a seqlen x outputsize [x ...] sequence tensor
-- Applies a forward RNN to each element in the sequence in
-- forward order and applies a backward RNN in reverse order.
-- For each step, the outputs of both RNNs are merged together using
-- the merge module (defaults to nn.CAddTable()).
------------------------------------------------------------------------
local BiSequencer, parent = torch.class('nn.BiSequencer', 'nn.AbstractSequencer')

function BiSequencer:__init(forward, backward, merge)
   parent.__init(self)

   if not torch.isTypeOf(forward, 'nn.Module') then
      error"BiSequencer: expecting nn.Module instance at arg 1"
   end

   if not backward then
      backward = forward:clone()
      backward:reset()
   end
   if not torch.isTypeOf(backward, 'nn.Module') then
      error"BiSequencer: expecting nn.Module instance or nil at arg 2"
   end

   -- for table sequences use nn.Sequential():add(nn.ZipTable()):add(nn.Sequencer(nn.JoinTable(1,1)))
   merge = merge or nn.CAddTable()
   if not torch.isTypeOf(merge, 'nn.Module') then
      error"BiSequencer: expecting nn.Module instance or nil at arg 3"
   end

   -- make into sequencer (if not already the case)
   forward = self.isSeq(forward) and forward or nn.Sequencer(forward)
   backward = self.isSeq(backward) and backward or nn.Sequencer(backward)

   -- the backward sequence reads the input in reverse and outputs the output in correct order
   backward = nn.ReverseUnreverse(backward)

   local brnn = nn.Sequential()
      :add(nn.ConcatTable():add(forward):add(backward))
      :add(merge)

   -- so that it can be handled like a Container
   self.modules[1] = brnn
end

-- forward RNN can remember. backward RNN can't.
function BiSequencer:remember(remember)
   local fwd, bwd = self:getForward(), self:getBackward()
   fwd:remember(remember)
   bwd:remember('neither')
   return self
end

function BiSequencer.isSeq(module)
   return torch.isTypeOf(module, 'nn.AbstractSequencer') or torch.typename(module):find('nn.Seq.+')
end

-- multiple-inheritance
nn.Decorator.decorate(BiSequencer)

function BiSequencer:getForward()
   return self:get(1):get(1):get(1)
end

function BiSequencer:getBackward()
   return self:get(1):get(1):get(2):getModule()
end

function BiSequencer:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'forward:  ' .. tostring(self:getForward()):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'backward: ' .. tostring(self:getBackward()):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'merge:    ' .. tostring(self:get(1):get(2)):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end