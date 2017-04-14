
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