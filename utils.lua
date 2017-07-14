
-- returns a buffer table local to a thread (no serialized)
function torch.getBufferTable(namespace)
   assert(torch.type(namespace) == 'string')
   torch._buffer = torch._buffer or {}
   torch._buffer[namespace] = torch._buffer[namespace] or {}
   return torch._buffer[namespace]
end

function torch.getBuffer(namespace, buffername, classname)
   local buffertable = torch.getBufferTable(namespace)
   assert(torch.type(buffername) == 'string')
   local buffer = buffertable[buffername]
   classname = (torch.type(classname) == 'string') and classname or torch.type(classname)

   if buffer then
      if torch.type(buffer) ~= classname then
  	     buffer = torch.factory(classname)()
  	     buffertable[buffername] = buffer
  	  end
   else
  	  buffer = torch.factory(classname)()
  	  buffertable[buffername] = buffer
   end

   return buffer
end

function torch.isByteTensor(tensor)
   local typename = torch.typename(tensor)
   if typename and typename:find('torch.*ByteTensor') then
      return true
   end
   return false
end

function torch.isCudaTensor(tensor)
   local typename = torch.typename(tensor)
   if typename and typename:find('torch.Cuda*Tensor') then
      return true
   end
   return false
end

function nn.utils.getZeroMaskBatch(batch, zeroMask)
   -- get first tensor
   local first = nn.utils.recursiveGetFirst(batch)
   first = first:contiguous():view(first:size(1), -1) -- collapse non-batch dims

   -- build mask (1 where norm is 0 in first)
   local _zeroMask = torch.getBuffer('getZeroMaskBatch', '_zeroMask', first)
   _zeroMask:norm(first, 2, 2)
   zeroMask = zeroMask or (
       (torch.type(first) == 'torch.CudaTensor') and torch.CudaByteTensor()
       or (torch.type(first) == 'torch.ClTensor') and torch.ClTensor()
       or torch.ByteTensor()
    )
   _zeroMask.eq(zeroMask, _zeroMask, 0)
   return zeroMask:view(zeroMask:size(1))
end

function nn.utils.getZeroMaskSequence(sequence, zeroMask)
   assert(torch.isTensor(sequence), "nn.utils.getZeroMaskSequence expecting tensor for arg 1")
   assert(sequence:dim() >= 2, "nn.utils.getZeroMaskSequence expecting seqlen x batchsize [x ...] tensor for arg 1")

   sequence = sequence:contiguous():view(sequence:size(1), sequence:size(2), -1)
   -- build mask (1 where norm is 0 in first)
   local _zeroMask
   if sequence.norm then
      _zeroMask = torch.getBuffer('getZeroMaskSequence', '_zeroMask', sequence)
   else
      _zeroMask = torch.getBuffer('getZeroMaskSequence', '_zeroMask', 'torch.FloatTensor')
      local _sequence = torch.getBuffer('getZeroMaskSequence', '_sequence', 'torch.FloatTensor')
      _sequence:resize(sequence:size()):copy(sequence)
      sequence = _sequence
   end
   _zeroMask:norm(sequence, 2, 3)

   zeroMask = zeroMask or torch.isCudaTensor(sequence) and torch.CudaByteTensor() or torch.ByteTensor()
   _zeroMask.eq(zeroMask, _zeroMask, 0)
   return zeroMask:view(sequence:size(1), sequence:size(2))
end

function nn.utils.recursiveSet(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveSet(t1[key], t2[key])
      end
      for i=#t2+1,#t1 do
         t1[i] = nil
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:set(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function nn.utils.recursiveTensorEq(t1, t2)
   if torch.type(t2) == 'table' then
      local isEqual = true
      if torch.type(t1) ~= 'table' then
         return false
      end
      for key,_ in pairs(t2) do
          isEqual = isEqual and nn.utils.recursiveTensorEq(t1[key], t2[key])
      end
      return isEqual
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      local diff = t1-t2
      local err = diff:abs():max()
      return err < 0.00001
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
end

function nn.utils.recursiveNormal(t2)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = nn.utils.recursiveNormal(t2[key])
      end
   elseif torch.isTensor(t2) then
      t2:normal()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function nn.utils.recursiveSum(t2)
   local sum = 0
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         sum = sum + nn.utils.recursiveSum(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      return t2:sum()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return sum
end

function nn.utils.recursiveNew(t2)
   if torch.type(t2) == 'table' then
      local t1 = {}
      for key,_ in pairs(t2) do
         t1[key] = nn.utils.recursiveNew(t2[key])
      end
      return t1
   elseif torch.isTensor(t2) then
      return t2.new()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
end

function nn.utils.recursiveGetFirst(input)
   if torch.type(input) == 'table' then
      return nn.utils.recursiveGetFirst(input[1])
   else
      assert(torch.isTensor(input))
      return input
   end
end

-- in-place set tensor to zero where zeroMask is 1
function nn.utils.recursiveZeroMask(tensor, zeroMask)
   if torch.type(tensor) == 'table' then
      for k,tensor_k in ipairs(tensor) do
         nn.utils.recursiveZeroMask(tensor_k, zeroMask)
      end
   else
      assert(torch.isTensor(tensor))

      local tensorSize = tensor:size():fill(1)
      tensorSize[1] = tensor:size(1)
      if zeroMask:dim() == 2 then
         tensorSize[2] = tensor:size(2)
      end
      assert(zeroMask:dim() <= tensor:dim())
      zeroMask = zeroMask:view(tensorSize):expandAs(tensor)
      -- set tensor to zero where zeroMask is 1
      tensor:maskedFill(zeroMask, 0)
   end
   return tensor
end

function nn.utils.recursiveDiv(tensor, scalar)
   if torch.type(tensor) == 'table' then
      for j=1,#tensor do
         nn.utils.recursiveDiv(tensor[j], scalar)
      end
   else
      tensor:div(scalar)
   end
end

function nn.utils.recursiveIndex(dst, src, dim, indices)
   if torch.type(src) == 'table' then
      dst = torch.type(dst) == 'table' and dst or {}
      for k,v in ipairs(src) do
         dst[k] = nn.utils.recursiveIndex(dst[k], v, dim, indices)
      end
      for i=#src+1,#dst do
         dst[i] = nil
      end
   else
      assert(torch.isTensor(src))
      dst = torch.isTensor(dst) and dst or src.new()

      dst:index(src, dim, indices)
   end
   return dst
end
nn.utils.recursiveIndexSelect = nn.utils.recursiveIndex

function nn.utils.recursiveIndexCopy(dst, dim, indices, src)
   if torch.type(src) == 'table' then
      dst = (torch.type(dst) == 'table') and dst or {dst}
      for key,src_ in pairs(src) do
         dst[key] = nn.utils.recursiveIndexCopy(dst[key], dim, indices, src_)
      end
      for i=#src+1,#dst do
         dst[i] = nil
      end
   elseif torch.isTensor(src) then
      assert(torch.isTensor(dst))
      dst:indexCopy(dim, indices, src)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(dst).." and "..torch.type(src).." instead")
   end
   return dst
end

function nn.utils.recursiveMaskedSelect(dst, src, mask)
   if torch.type(src) == 'table' then
      dst = (torch.type(dst) == 'table') and dst or {dst}
      for key,src_ in pairs(src) do
         dst[key] = nn.utils.recursiveMaskedSelect(dst[key], src_, mask)
      end
      for i=#src+1,#dst do
         dst[i] = nil
      end
   elseif torch.isTensor(src) then
      assert(torch.isTensor(dst))
      dst:maskedSelect(src, mask)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(dst).." and "..torch.type(src).." instead")
   end
   return dst
end

function nn.utils.recursiveMaskedCopy(dst, mask, src)
   if torch.type(src) == 'table' then
      dst = (torch.type(dst) == 'table') and dst or {dst}
      for key,src_ in pairs(src) do
         dst[key] = nn.utils.recursiveMaskedCopy(dst[key], mask, src_)
      end
      for i=#src+1,#dst do
         dst[i] = nil
      end
   elseif torch.isTensor(src) then
      assert(torch.isTensor(dst))
      dst:maskedCopy(mask, src)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(dst).." and "..torch.type(src).." instead")
   end
   return dst
end

function nn.utils.setZeroMask(modules, zeroMask, cuda)
   if cuda and not torch.isCudaTensor(zeroMask) then
      cuZeroMask = torch.getBuffer('setZeroMask', 'cuZeroMask', 'torch.CudaByteTensor')
      cuZeroMask:resize(zeroMask:size()):copy(zeroMask)
      zeroMask = cuZeroMask
   end
   for i,module in ipairs(torch.type(modules) == 'table' and modules or {modules}) do
      module:setZeroMask(zeroMask)
   end
end
function nn.utils.get_ngrams(sent, n, count)
   local ngrams = {}
   for beg = 1, #sent do
      for  last= beg, math.min(beg+n-1, #sent) do
         local ngram = table.concat(sent, ' ', beg, last)
	 local len = last-beg+1 -- keep track of ngram length
         if not count then
            ngrams[ngram] = 1
         else
            if ngrams[ngram] == nil then
               ngrams[ngram] = {1, len}
            else
               ngrams[ngram][1] = ngrams[ngram][1] + 1
            end
         end
      end
   end
   return ngrams
end

function nn.utils.get_skip_bigrams(sent, ref, count, dskip)
   local skip_bigrams = {}
   ref = ref or sent
   for beg = 1, #sent do
      if ref[sent[beg]] then
	 local temp_token = sent[beg]
	 for  last= beg+1, math.min(beg + dskip-1, #sent) do
	    if ref[sent[last]] then
	       skip_bigram = temp_token..sent[last]
	       if not count then
		  skip_bigrams[skip_bigram] = 1
	       else
		  skip_bigrams[skip_bigram] = (skip_bigram[bigram] or 0) + 1
	       end
	    end
	 end
      end
   end
   return skip_bigrams
end


function nn.utils.get_ngram_prec(cand, ref, n)
   local results = {}
   for i = 1, n do
      results[i] = {0, 0}
   end
   local cand_ngrams = nn.utils.get_ngrams(cand, n, 1)
   local ref_ngrams = nn.utils.get_ngrams(ref, n, 1)
   for ngram, dist in pairs(cand_ngrams) do
      local freq = dist[1]
      local length = dist[2]
      results[length][1] = results[length][1] + freq
      local actual
      if ref_ngrams[ngram] == nil then
         actual = 0
      else
         actual = ref_ngrams[ngram][1]
      end
      results[length][2] = results[length][2] + math.min(actual, freq)
   end
   return results
end

function nn.utils.get_ngram_recall(cand, ref, n)
   local results = {}
   for i = 1, n do
      results[i] = {0, 0}
   end
   local cand_ngrams = nn.utils.get_ngrams(cand, n, 1)
   local ref_ngrams = nn.utils.get_ngrams(ref, n, 1)
   for ngram, dist in pairs(ref_ngrams) do
      local freq = dist[1]
      local length = dist[2]
      results[length][1] = results[length][1] + freq
      local actual
      if cand_ngrams[ngram] == nil then
         actual = 0
      else
         actual = cand_ngrams[ngram][1]
      end
      results[length][2] = results[length][2] + math.min(actual, freq)
   end
   return results
end
