require 'rnn'

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setdefaulttensortype('torch.DoubleTensor')
local timesteps = 16
local bsize = 16
local embeddingSize = 128
local outputSize = 128
local ntests = 100
local sizes = torch.LongTensor(timesteps):fill(bsize)
-- sizes[1] = bsize
-- for i=1,timesteps do
--    sizes[i] = math.max(bsize - i + 1,1)
-- end
local input0 = torch.Tensor(timesteps*512*embeddingSize):uniform()
local gradOutput0 = torch.Tensor(timesteps*512*embeddingSize):uniform()
local input = torch.Tensor(timesteps, bsize, embeddingSize)
local mask = torch.ByteTensor(timesteps, bsize):zero()
local gradOutput = torch.Tensor(timesteps, bsize, outputSize)
for i=1,bsize do
   input[{{},{i,i},{}}]:copy(input0[{{timesteps*(i-1)*embeddingSize+1,timesteps*i*embeddingSize}}])
   gradOutput[{{},{i,i},{}}]:copy(gradOutput0[{{timesteps*(i-1)*outputSize+1,timesteps*i*outputSize}}])
end
local input_flat = torch.Tensor(sizes:sum(), embeddingSize)
local gradOutput_flat = torch.Tensor(sizes:sum(), embeddingSize)

local runningIdx = 1
for i=1,timesteps do
   input_flat[{{runningIdx,runningIdx+sizes[i]-1},{}}]:copy(input[{{i,i},{1,sizes[i]},{}}])
   gradOutput_flat[{{runningIdx,runningIdx+sizes[i]-1},{}}]:copy(gradOutput[{{i,i},{1,sizes[i]},{}}])
   if sizes[i] < bsize then
--       input[{{i,i},{sizes[i]+1,bsize},{}}]:zero()
--       gradOutput[{{i,i},{sizes[i]+1,bsize},{}}]:zero()
      mask[{{i,i},{sizes[i]+1,bsize}}]:fill(1)
   end
   runningIdx = runningIdx + sizes[i]
end
local seqLSTM = nn.SeqLSTM(embeddingSize, outputSize)
seqLSTM:maskZero()
-- seqLSTM:remember('both')

local output_flat = torch.Tensor()
local gradInput_flat = torch.Tensor()

local weight = seqLSTM.weight
local gradWeight = seqLSTM.gradWeight
local gradBias = seqLSTM.gradBias
local bias = seqLSTM.bias
local inputC = torch.Tensor(bsize, outputSize):zero() --:uniform()
local inputH = torch.Tensor(bsize, outputSize):zero() --:uniform()
local gradInputC = torch.Tensor(bsize, outputSize):zero() --:uniform()
local gradInputH = torch.Tensor(bsize, outputSize):zero() -- :uniform()
local buffer = torch.Tensor()
local gradInputBuffer = torch.Tensor()

local output
sys.tic()
for i=1,ntests do
--    seqLSTM.h0 = inputH
--    seqLSTM.c0 = inputC
--    seqLSTM.grad_hT = gradInputH
--    seqLSTM.grad_cT = gradInputC
   seqLSTM:setZeroMask(mask)
   output = seqLSTM:forward(input)
   seqLSTM:setZeroMask(mask)
   gradInput = seqLSTM:backward(input, gradOutput)
end
local seqLSTMDuration = sys.toc()
print('SeqLSTM: ', seqLSTMDuration)
print('gradInput:mean(): ', gradInput:mean())

sys.tic()
inputC:zero()
inputH:zero()
gradInputC:zero()
gradInputH:zero()
gradWeight:zero()
for i=1,ntests do
   input.THRNN.LSTM_updateOutput(
             input_flat:cdata(),
             inputC:cdata(),
             inputH:cdata(),
             sizes:cdata(),
             output_flat:cdata(),
             weight:cdata(),
             bias:cdata(),
             buffer:cdata(),
             embeddingSize,
             outputSize,
             1,
             0,
             0)
   input.THRNN.LSTM_backward(
             input_flat:cdata(),
             inputC:cdata(),
             inputH:cdata(),
             sizes:cdata(),
             gradOutput_flat:cdata(),
             gradInput_flat:cdata(),
             gradInputH:cdata(),
             gradInputC:cdata(),
             weight:cdata(),
             bias:cdata(),
             buffer:cdata(),
             gradInputBuffer:cdata(),
             gradWeight:cdata(),
             gradBias:cdata(),
             output_flat:cdata(),
             1,
             0,
             embeddingSize,
             outputSize,
             1,
             0)
end
-- print('gradInput_flat: ', gradInput_flat)
local CLSTMDuration = sys.toc()
print('CLSTM: ', CLSTMDuration)
print('Average CLSTM (us): ', CLSTMDuration / ntests * 1e6)
print('speedup: ' .. (seqLSTMDuration / CLSTMDuration))
print('gradWeight:mean(): ', gradWeight:mean())
print('gradInput_flat:mean(): ', gradInput_flat:mean())

print{output, output_flat}
print{
   input_flat = input_flat,
   sizes = sizes,
   output_flat = output_flat,
   weight = weight,
   bias = bias,
   inputC = inputC,
   inputH = inputH,
   buffer = buffer,
}
print('compare outputs')
local runningIdx = 1
for i=1,timesteps do
   print(torch.add(output[{{i,i},{1,sizes[i]},{}}], -1, output_flat[{{runningIdx,runningIdx+sizes[i]-1},{}}]):abs():max())
   runningIdx = runningIdx + sizes[i]
end
print('')
print('compare gradInputs')
runningIdx = 1
for i=1,timesteps do
   print(torch.add(gradInput[{{i,i},{1,sizes[i]},{}}], -1, gradInput_flat[{{runningIdx,runningIdx+sizes[i]-1},{}}]):abs():max())
   runningIdx = runningIdx + sizes[i]
end
-- print(torch.add(output, -1, output_flat:view(timesteps, bsize, outputSize)):abs():max())
