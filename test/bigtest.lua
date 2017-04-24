local _ = require 'moses'
local rnnbigtest = {}
local precision = 1e-5
local mytester

function rnnbigtest.NCE_nan()
   local success, dl = pcall(function() return require 'dataload' end)
   if not success then
      return
   end
   if not pcall(function() require 'cunn' end) then
      return
   end

   local datapath = paths.concat(dl.DATA_PATH, 'BillionWords')
   local wordfreq = torch.load(paths.concat(datapath, 'word_freq.th7'))
   local unigram = wordfreq:float()--:add(0.0000001):log()
   print("U", unigram:min(), unigram:mean(), unigram:std(), unigram:max())

   local batchsize = 128
   local seqlen = 50
   local hiddensize = 200
   local vocabsize = unigram:size(1)
   local k = 400

   local tinyset = dl.MultiSequenceGBW(datapath, 'train_tiny.th7', batchsize, verbose)

   local lm = nn.Sequential()
   local lookup = nn.LookupTableMaskZero(vocabsize, hiddensize)
   lm:add(lookup)

   for i=1,2 do
      local rnn = nn.SeqLSTM(hiddensize, hiddensize)
      rnn.maskzero = true
      lm:add(rnn)
   end

   lm:add(nn.SplitTable(1))

   local ncemodule = nn.NCEModule(hiddensize, vocabsize, k, unigram, 1)

   lm = nn.Sequential()
      :add(nn.ParallelTable()
         :add(lm):add(nn.Identity()))
      :add(nn.ZipTable())

   lm:add(nn.Sequencer(nn.MaskZero(ncemodule, 1)))
   lm:remember()

   local crit = nn.MaskZeroCriterion(nn.NCECriterion(), 0)
   local targetmodule = nn.Sequential():add(nn.Convert()):add(nn.SplitTable(1))
   local criterion = nn.SequencerCriterion(crit)

   for k,param in ipairs(lm:parameters()) do
      param:uniform(-0.1, 0.1)
   end

   -- comment this out to see the difference
   ncemodule:reset()

   lm:training()

   lm:cuda()
   criterion:cuda()
   targetmodule:cuda()

   local sumErr = 0
   local _ = require 'moses'
   for k,inputs, targets in tinyset:subiter(seqlen, 512) do
      local targets = targetmodule:forward(targets)
      local inputs = {inputs, targets}
      -- forward
      local outputs = lm:forward(inputs)
      for i,output in ipairs(outputs) do
         assert(not _.isNaN(output[1]:sum()), tostring(i))
         assert(not _.isNaN(output[2]:sum()), tostring(i))
         assert(not _.isNaN(output[3]:sum()), tostring(i))
         assert(not _.isNaN(output[4]:sum()), tostring(i))
      end
      local err = criterion:forward(outputs, targets)
      assert(not _.isNaN(err))
      sumErr = sumErr + err
      -- backward
      local gradOutputs = criterion:backward(outputs, targets)

      for i,gradOutput in ipairs(gradOutputs) do
         assert(not _.isNaN(gradOutput[1]:sum()), tostring(i))
         assert(not _.isNaN(gradOutput[2]:sum()), tostring(i))
      end
      lm:zeroGradParameters()
      lm:backward(inputs, gradOutputs)
      lm:updateParameters(0.7)
      local params, gradParams = lm:parameters()

      for i,param in ipairs(params) do
         assert(not _.isNaN(param:sum()), tostring(i))
         assert(not _.isNaN(gradParams[i]:sum()), tostring(i))
      end

      local counts = {}
      inputs[1]:float():apply(function(x)
         counts[x] = (counts[x] or 0) + 1
      end)

      print("Top freqs", unpack(_.last(_.sort(_.values(counts)), 5)))
      print("Batch : "..k..", err="..err)
      for name,module in pairs{LT=lookup, NCE=ncemodule} do
         print(name..".gradWeight : "..module.gradWeight:norm()..", .weight : "..module.weight:norm())
         if name == 'NCE' then
            print(name..".gradBias : "..module.gradBias:norm()..", .bias : "..module.bias:norm())
         end
      end
   end

end

function rnnbigtest.Reinforce()
   -- let us try to reinforce an mlp to learn a simple distribution
   local n = 10
   local inputs = torch.Tensor(n,3):uniform(0,0.1)
   local targets = torch.Tensor(n):fill(0)
   local stdev = 0.5
   local beta = 0.9
   local alpha = 1
   local lr = 0.1

   for i=1,inputs:size(1) do
      local j = (i % inputs:size(2)) + 1
      inputs[{i,j}] = torch.uniform(0.9,1.1)
      targets[i] = j
   end

   local M = 10
   local function train(mlp, cost, N, name)
      local converged = false
      local baseReward
      local reward
      for i=1,M do
         mlp:reset()

         baseReward = 0
         for i=1,inputs:size(1) do
            mlp:evaluate()
            local target = targets:narrow(1,i,1)
            local output = mlp:forward(inputs:narrow(1,i,1))
            baseReward = baseReward - cost:forward(output, target)
         end
         baseReward = baseReward/inputs:size(1)

         for k=1,N do

            for i=1,inputs:size(1) do
               mlp:training()
               mlp:zeroGradParameters()
               local target = targets:narrow(1,i,1)
               local output = mlp:forward(inputs:narrow(1,i,1))
               local err = cost:forward(output, target)
               local gradOutput = cost:backward(output, target)
               mlp:backward(inputs:narrow(1,i,1), gradOutput)
               mlp:updateParameters(lr)
            end

            reward = 0
            for i=1,inputs:size(1) do
               mlp:evaluate()
               local target = targets:narrow(1,i,1)
               local output = mlp:forward(inputs:narrow(1,i,1))
               reward = reward - cost:forward(output, target)
            end
            reward = reward/inputs:size(1)

            -- is the baseReward lesser than 70% of reward after training?
            -- i.e. did the reward increase sufficiently?
            if reward*0.7 > baseReward then
               converged = true
               break
            end
         end

         if reward*0.7 > baseReward then
            converged = true
            break
         end
      end

      mytester:assert(converged, name.." did not converge : "..reward.."*0.7 < "..baseReward)
   end

   -- ReinforceNormal
   local hiddenSize = 200
   local N = 10
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Tanh())
   mlp:add(nn.ReinforceNormal(stdev))
   mlp:add(nn.Clip(-1,1))
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())

   local concat = nn.ConcatTable()
   concat:add(mlp)
   concat:add( nn.Sequential():add( nn.Constant(1,1) ):add(nn.Add(1)) )

   local cost = nn.VRClassReward(concat, alpha)

   train(concat, cost, N, 'ReinforceNormal')

   -- ReinforceGamma
   local hiddenSize = 200
   local N = 10
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Sigmoid())
   mlp:add(nn.ReinforceGamma(stdev))
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())

   local concat = nn.ConcatTable()
   concat:add(mlp)
   concat:add( nn.Sequential():add( nn.Constant(1,1) ):add(nn.Add(1)) )

   local cost = nn.VRClassReward(concat, alpha)

   train(concat, cost, N, 'ReinforceGamma')

   -- ReinforceBernoulli
   local hiddenSize = 20
   local N = 30
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Sigmoid())
   mlp:add(nn.ReinforceBernoulli())
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())

   local concat = nn.ConcatTable()
   concat:add(mlp)
   concat:add( nn.Sequential():add( nn.Constant(1,1) ):add(nn.Add(1)) )

   local cost = nn.VRClassReward(concat, alpha)

   train(concat, cost, N, 'ReinforceBernoulli')

   -- ReinforceCategorical
   local hiddenSize = 200
   local N = 10
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())
   mlp:add(nn.AddConstant(0.00001))
   mlp:add(nn.ReinforceCategorical())

   local concat = nn.ConcatTable()
   concat:add(mlp)
   concat:add( nn.Sequential():add( nn.Constant(1,1) ):add(nn.Add(1)) )

   local cost = nn.VRClassReward(concat, alpha)

   train(concat, cost, N, 'ReinforceCategorical')
end

-- Unit Test Kmeans layer
function rnnbigtest.Kmeans()
   local k = 10
   local dim = 5
   local batchSize = 1000
   local input = torch.rand(batchSize, dim)
   for i=1, batchSize do
      input[i]:fill(torch.random(1, k))
   end

   local verbose = false

   local attempts = 10
   local iter = 100
   local bestLoss = 100000000
   local bestKm = nil
   local tempLoss = 0
   local learningRate = 1

   local initTypes = {'random', 'kmeans++'}
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   for _, initType in pairs(initTypes) do
      for _, useCuda in pairs(useCudas) do

         sys.tic()
         for j=1, attempts do
            local km = nn.Kmeans(k, dim)

            if initType == 'kmeans++' then
               km:initKmeansPlus(input)
            else
               km:initRandom(input)
            end

            if useCuda then km:cuda() end
            for i=1, iter do
               km:zeroGradParameters()

               km:forward(input)
               km:backward(input, gradOutput)

               -- Gradient descent
               km.weight:add(-learningRate, km.gradWeight)
               tempLoss = km.loss
            end
            if verbose then print("Attempt Loss " .. j ..": " .. tempLoss) end
            if tempLoss < bestLoss then
               bestLoss = tempLoss
            end
         end
         if verbose then
            print("InitType: " .. initType .. " useCuda: " .. tostring(useCuda))
            print("Best Loss: " .. bestLoss)
            print("Total time: " .. sys.toc())
         end
         if initType == 'kmeans++' then
            mytester:assert(bestLoss < 0.00001)
         else
            mytester:assert(bestLoss < 500)
         end
      end
   end
end

function rnnbigtest.NCE_benchmark()
   pcall(function() require 'cunn' end) -- make sure to import cunn before initializing large tensors, else weird segfault...

   local nclass = 1000000
   local hiddensize = 200
   local batchsize = 50
   local nloop = 5
   local k = 25
   local unigrams = torch.Tensor(nclass):uniform(0,1)
   local mlp = nn.Sequential()
      :add(nn.Linear(hiddensize, nclass))
      :add(nn.SoftMax())
   local nll = nn.ClassNLLCriterion()

   local nce = nn.NCEModule(hiddensize, nclass, 25, unigrams)
   local crit = nn.NCECriterion()

   local input = torch.randn(batchsize, hiddensize)
   local target = torch.LongTensor(batchsize):random(1,nclass)

   local sync = function() return end
   if pcall(function() require 'cunn' end) then
      input = input:cuda()
      target = target:cuda()
      nce:cuda()
      crit:cuda()
      mlp:cuda()
      nll:cuda()
      sync = function() cutorch.synchronize() end
   end

   local output = nce:forward{input, target}
   local loss = crit:forward(output, target)
   local gradOutput = crit:backward(output, target)
   local gradInput = nce:backward({input, target}, gradOutput)

   local output = mlp:forward(input)
   local loss = nll:forward(output, target)
   local gradOutput = nll:backward(output, target)
   local gradInput = mlp:backward(input, gradOutput)

   sync()
   local a = torch.Timer()
   for i=1,nloop do
      output = nce:forward{input, target}
   end
   sync()
   local ncefwd = a:time().real

   a:reset()
   for i=1,nloop do
      loss = crit:forward(output, target)
   end
   sync()
   local critfwd = a:time().real

   a:reset()
   for i=1,nloop do
      gradOutput = crit:backward(output, target)
   end
   sync()
   local critbwd = a:time().real

   a:reset()
   for i=1,nloop do
      gradInput = nce:backward({input, target}, gradOutput)
   end
   sync()
   local ncebwd = a:time().real

   -- mlp nll
   local a = torch.Timer()
   for i=1,nloop do
      output = mlp:forward(input)
   end
   sync()
   local mlpfwd = a:time().real

   a:reset()
   for i=1,nloop do
      loss = nll:forward(output, target)
   end
   sync()
   local nllfwd = a:time().real

   a:reset()
   for i=1,nloop do
      gradOutput = nll:backward(output, target)
   end
   sync()
   local nllbwd = a:time().real

   a:reset()
   for i=1,nloop do
      gradInput = mlp:backward(input, gradOutput)
   end
   sync()
   local mlpbwd = a:time().real

   local ncetotal = ncefwd+critfwd+critbwd+ncebwd
   local lintotal = mlpfwd+nllfwd+nllbwd+mlpbwd
   print("module:forward (nce vs linear)", ncefwd, mlpfwd)
   print("criterion:forward (nce vs nll)", critfwd, nllfwd)
   print("criterion:backward (nce vs nll)", critbwd, nllbwd)
   print("module:backward (nce vs linear)", ncebwd, mlpbwd)
   print("total (nce vs linear)", ncetotal, lintotal, lintotal/ncetotal)

   if not (cunn and cutorch.getDeviceCount() > 1) then
      return
   end

   nce:multicuda(1,2)

   local output = nce:forward{input, target}
   local loss = crit:forward(output, target)
   local gradOutput = crit:backward(output, target)
   local gradInput = nce:backward({input, target}, gradOutput)
   sync()

   local a = torch.Timer()
   for i=1,nloop do
      output = nce:forward{input, target}
   end
   sync()
   local ncefwd2 = a:time().real

   a:reset()
   for i=1,nloop do
      gradInput = nce:backward({input, target}, gradOutput)
   end
   sync()
   local ncebwd2 = a:time().real

   local total1 = ncefwd+ncebwd
   local total2 = ncefwd2+ncebwd2
   print("module:forward (1 vs 2 gpu)", ncefwd, ncefwd2)
   print("module:backward (1 vs 2 gpu)", ncebwd, ncebwd2)
   print("total (1 vs 2 gpu)", total1, total2, total2/total1)
end


function rnnbigtest.LSTM()
   local seqlen, batchsize = 30, 32
   local inputsize, outputsize = 128, 128
   local nloop = 20

   local lstms = {}
   lstms.fast = nn.Sequencer(nn.FastLSTM(inputsize, outputsize))
   local stepmodule = nn.Sequential()
      :add(nn.FlattenTable())
      :add(nn.StepLSTM(inputsize, outputsize))
   local recmodule = nn.Sequential()
      :add(nn.Recurrence(stepmodule, {{outputsize}, {outputsize}}, 1, seqlen))
      :add(nn.SelectTable(1))
   lstms.step = nn.Sequencer(recmodule)
   lstms.rec = nn.Sequencer(nn.RecLSTM(inputsize, outputsize))
   local luarec = nn.RecLSTM(inputsize, outputsize)
   luarec.modules[1].forceLua = true
   lstms.luarec = nn.Sequencer(luarec)
   lstms.seq = nn.SeqLSTM(inputsize, outputsize)

   local input = torch.Tensor(seqlen, batchsize, inputsize)
   local gradOutput = torch.Tensor(seqlen, batchsize, outputsize)

   local t = torch.Timer()

   print("CPU test")

   for name, lstm in pairs(lstms) do
       -- warmup
      lstm:remember('neither')
      lstm:forward(input)
      lstm:zeroGradParameters()
      lstm:backward(input, gradOutput)
      -- main test
      t:reset()
      for i=1,nloop do
         lstm:forward(input)
         lstm:zeroGradParameters()
         lstm:backward(input, gradOutput)
      end
      lstm.testtime = t:time().real/10
   end

   for i,name in ipairs{'fast','step','luarec','rec','seq'} do
      print(name..' LSTM time: '..lstms[name].testtime..' seconds')
   end

   print("RecLSTM-C "..(lstms.luarec.testtime/lstms.rec.testtime)..' faster than RecLSTM-Lua')
   print("RecLSTM "..(lstms.fast.testtime/lstms.rec.testtime)..' faster than FastLSTM')
   print("SeqLSTM "..(lstms.rec.testtime/lstms.seq.testtime)..' faster than RecLSTM')

   print("Memory test")

   for name, lstm in pairs(lstms) do
      lstm.fullsize = #torch.serialize(lstm)
      lstm:clearState()
      lstm.clearsize = #torch.serialize(lstm)
   end

   for i,name in ipairs{'fast','step','rec','seq'} do
      print(name..' LSTM memory: '..lstms[name].fullsize/(1024*1024)..':'..lstms[name].clearsize/(1024*1024)..' MB')
   end
end


function rnn.bigtest(tests)
   mytester = torch.Tester()
   mytester:add(rnnbigtest)
   math.randomseed(os.time())
   mytester:run(tests)
end
