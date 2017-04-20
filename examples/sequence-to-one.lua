require 'rnn'

-- hyper-parameters
batchSize = 8
seqlen = 10 -- sequence length
hiddenSize = 100
nIndex = 100 -- input words
nClass = 7 -- output classes
lr = 0.1


-- build simple recurrent neural network
rec = nn.Recurrence(
   nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Identity())
         :add(nn.Linear(hiddenSize, hiddenSize)))
      :add(nn.CAddTable())
      :add(nn.Sigmoid())
   , hiddenSize, 1)

rnn = nn.Sequential()
   :add(nn.LookupTable(nIndex, hiddenSize))
   :add(nn.SplitTable(1,2))
   :add(nn.Sequencer(rec))
   :add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
   :add(nn.Linear(hiddenSize, nClass))
   :add(nn.LogSoftMax())

-- build criterion

criterion = nn.ClassNLLCriterion()

-- build dummy dataset (task is to predict class given seqlen words)
-- similar to sentiment analysis datasets
ds = {}
ds.size = 1000
ds.input = torch.LongTensor(ds.size,seqlen)
ds.target = torch.LongTensor(ds.size):random(nClass)

-- this will make the inputs somewhat correlate with the targets,
-- such that the reduction in training error should be more obvious
local correlate = torch.LongTensor(nClass, seqlen*3):random(nClass)
local indices = torch.LongTensor(seqlen)
local buffer = torch.LongTensor()
local sortVal, sortIdx = torch.LongTensor(), torch.LongTensor()
for i=1,ds.size do
   indices:random(1,seqlen*3)
   buffer:index(correlate[ds.target[i]], 1, indices)
   sortVal:sort(sortIdx, buffer, 1)
   ds.input[i]:copy(sortVal:view(-1))
end


indices:resize(batchSize)

-- training
local inputs, targets = torch.LongTensor(), torch.LongTensor()
for iteration = 1, 1000 do
   -- 1. create a sequence of seqlen time-steps

   indices:random(1,ds.size) -- choose some random samples
   inputs:index(ds.input, 1,indices)
   targets:index(ds.target, 1,indices)

   -- 2. forward sequence through rnn

   rnn:zeroGradParameters()

   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)

   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)

   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)

   -- 4. update

   rnn:updateParameters(lr)
end
