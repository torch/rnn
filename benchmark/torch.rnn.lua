require('torch')
require('nn')
require('rnn')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('--mode', 'inference', 'inference or train')
cmd:option('--network_type', 'rnn', 'Network type (rnn, lstm)')
cmd:option('--hidden_size', 128, 'Neural network inputs and output size')
cmd:option('--seq_length', 16, 'Sequence length')
cmd:option('--batch_size', 32, 'Batch size')
cmd:option('--num_batches', 64, 'Number of samples')
cmd:option('--num_iter', 5, 'Number of iterations')
cmd:option('--force_lua', false, 'force use of Lua instead of C with LSTM')
cmd:option('--tensor_type', 'torch.FloatTensor', 'tensor type (to compare with TF test use torch.FloatTensor)')
cmd:text()

local args = cmd:parse(arg)
local nbatch = args.num_batches
local network = args.network_type
local hiddensize = args.hidden_size
local seqlen = args.seq_length
local batchsize = args.batch_size
local niter = args.num_iter
local nSamples = nbatch * batchsize
local tensor_type = args.tensor_type

local forward_only = args.mode:lower() == 'inference'
local inputs = torch.rand(nbatch, seqlen, batchsize, hiddensize):type(tensor_type)
local targets = torch.rand(nbatch, batchsize, hiddensize):type(tensor_type)

function BasicRNN(num_units)
   local rnn = nn.Sequential()
      :add(nn.JoinTable(1,1))
      :add(nn.Linear(num_units*2, num_units))
      :add(nn.Sigmoid())
   rnn = nn.Recurrence(rnn, num_units, 1)
   rnn = nn.Sequential()
      :add(nn.Sequencer(rnn))
      :add(nn.Select(1,-1))
   return rnn
end

local lstm
function BasicLSTM(num_units)
   lstm = nn.SeqLSTM(num_units, num_units)
   lstm.forceLua = args.force_lua
   rnn = nn.Sequential()
      :add(lstm)
      :add(nn.Select(1,-1))
   return rnn
end

local rnn
if network == 'rnn' then
   rnn = BasicRNN(hiddensize)
elseif network == 'lstm' then
   rnn = BasicLSTM(hiddensize)
else
   error('Unkown network type!')
end

local criterion = nn.MSECriterion()

rnn:type(tensor_type)
criterion:type(tensor_type)

local a = torch.Timer()

if infer then
   rnn:evaluate()
end

local infer = function(id)
   local out = rnn:forward(inputs[id])
end

local train = function(id)
   criterion:forward(rnn:forward(inputs[id]), targets[id])
   rnn:zeroGradParameters()
   rnn:backward(inputs[id], criterion:backward(rnn.output, targets[id]))
   rnn:updateParameters(0.01)
end

local func = forward_only and infer or train

-- Warmup
func(1)
local elapsed = 0
for j = 1, niter do
   a:reset()
   for i = 1, nbatch do
      if lstm then lstm.print_profile = (j == 1 and i == 1) end
      func(i)
   end
   local a_time = a:time().real
   print('iter ' .. j, a_time)
   elapsed = a_time + elapsed
end

elapsed = elapsed / niter
local samples_per_sec = nSamples / elapsed
print(args)
print(string.format('--- %d samples in %4.4f seconds (%4.2f samples/s, %4.4f ms/sample) ---',
                    nSamples, elapsed, samples_per_sec, 1000 / samples_per_sec))
