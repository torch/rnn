local ffi = require 'ffi'

local THRNN = {}


local generic_THRNN_h = require 'rnn.THRNN_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THRNN.h
generic_THRNN_h = generic_THRNN_h:gsub("\n#[^\n]*", "")
generic_THRNN_h = generic_THRNN_h:gsub("^#[^\n]*\n", "")

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void THRNNState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

-- polyfill for LUA 5.1
if not package.searchpath then
   local sep = package.config:sub(1,1)
   function package.searchpath(mod, path)
      mod = mod:gsub('%.', sep)
      for m in path:gmatch('[^;]+') do
         local nm = m:gsub('?', mod)
         local f = io.open(nm, 'r')
         if f then
            f:close()
            return nm
         end
     end
   end
end

-- load libTHRNN
THRNN.C = ffi.load(package.searchpath('libTHRNN', package.cpath))

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/THRNN/generic/THRNN.h
local preprocessed = string.gsub(generic_THRNN_h, 'TH_API void THRNN_%(([%a%d_]+)%)', 'void THRNN_TYPE%1')

local replacements =
{
   {
      ['TYPE'] = 'Double',
      ['real'] = 'double',
      ['THTensor'] = 'THDoubleTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
   },
   {
      ['TYPE'] = 'Float',
      ['real'] = 'float',
      ['THTensor'] = 'THFloatTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
    }
}

-- gsub(s, 'real', 'float') changes accreal to accfloat.
-- typedef accfloat ahead of time.
ffi.cdef("typedef double accfloat;")
-- gsub(s, 'real', 'double') changes accreal to accfloat.
-- typedef accdouble ahead of time
ffi.cdef("typedef double accdouble;")

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

THRNN.NULL = ffi.NULL or nil

function THRNN.getState()
   return ffi.NULL or nil
end

function THRNN.optionalTensor(t)
   return t and t:cdata() or THRNN.NULL
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THRNN_%(([%a%d_]+)%)') do
      t[#t+1] = n
   end
   return t
end

function THRNN.bind(lib, base_names, type_name, state_getter)
   local ftable = {}
   local prefix = 'THRNN_' .. type_name
   for i,n in ipairs(base_names) do
      -- use pcall since some libs might not support all functions (e.g. cunn)
      local ok,v = pcall(function() return lib[prefix .. n] end)
      if ok then
         ftable[n] = function(...) v(state_getter(), ...) end   -- implicitely add state
      else
         print('not found: ' .. prefix .. n .. v)
      end
   end
   return ftable
end

-- build function table
local function_names = extract_function_names(generic_THRNN_h)

THRNN.kernels = {}
THRNN.kernels['torch.FloatTensor'] = THRNN.bind(THRNN.C, function_names, 'Float', THRNN.getState)
THRNN.kernels['torch.DoubleTensor'] = THRNN.bind(THRNN.C, function_names, 'Double', THRNN.getState)

torch.getmetatable('torch.FloatTensor').THRNN = THRNN.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').THRNN = THRNN.kernels['torch.DoubleTensor']

function THRNN.runKernel(f, type, ...)
   local ftable = THRNN.kernels[type]
   if not ftable then
      error('Unsupported tensor type: '..type)
   end
   local f = ftable[f]
   if not f then
      error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
   end
   f(...)
end

return THRNN
