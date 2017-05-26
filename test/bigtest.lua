local _ = require 'moses'
local rnnbigtest = {}
local precision = 1e-5
local mytester


function rnnbigtest.LSTM_char_rnn()
   -- benchmark our LSTM against char-rnn's LSTM
   if not pcall(function() require 'cunn'; require 'nngraph' end) then
      return
   end

   local batch_size = 50
   local input_size = 65
   local rnn_size = 128
   local n_layer = 2
   local seq_len = 50

   local inputs = {}
   local gradOutputs = {}
   for i=1,seq_len do
      table.insert(inputs, torch.Tensor(batch_size):random(1,input_size):cuda())
      table.insert(gradOutputs, torch.randn(batch_size, input_size):cuda())
   end

   local a = torch.Timer()
   local function clone_list(tensor_list, zero_too)
       -- utility function. todo: move away to some utils file?
       -- takes a list of tensors and returns a list of cloned tensors
       local out = {}
       for k,v in pairs(tensor_list) do
           out[k] = v:clone()
           if zero_too then out[k]:zero() end
       end
       return out
   end

   local model_utils = {}
   function model_utils.combine_all_parameters(...)
      local con = nn.Container()
      for i, net in ipairs{...} do
         con:add(net)
      end
      return con:getParameters()
   end

   function model_utils.clone_many_times(net, T)
       local clones = {}

       local params, gradParams
       if net.parameters then
           params, gradParams = net:parameters()
           if params == nil then
               params = {}
           end
       end

       local paramsNoGrad
       if net.parametersNoGrad then
           paramsNoGrad = net:parametersNoGrad()
       end

       local mem = torch.MemoryFile("w"):binary()
       mem:writeObject(net)

       for t = 1, T do
           -- We need to use a new reader for each clone.
           -- We don't want to use the pointers to already read objects.
           local reader = torch.MemoryFile(mem:storage(), "r"):binary()
           local clone = reader:readObject()
           reader:close()

           if net.parameters then
               local cloneParams, cloneGradParams = clone:parameters()
               local cloneParamsNoGrad
               for i = 1, #params do
                   cloneParams[i]:set(params[i])
                   cloneGradParams[i]:set(gradParams[i])
               end
               if paramsNoGrad then
                   cloneParamsNoGrad = clone:parametersNoGrad()
                   for i =1,#paramsNoGrad do
                       cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                   end
               end
           end

           clones[t] = clone
           collectgarbage()
       end

       mem:close()
       return clones
   end

   local function makeCharLSTM(input_size, rnn_size, n)
      local dropout = 0

      -- there will be 2*n+1 inputs
      local inputs = {}
      table.insert(inputs, nn.Identity()()) -- x
      for L = 1,n do
         table.insert(inputs, nn.Identity()()) -- prev_c[L]
         table.insert(inputs, nn.Identity()()) -- prev_h[L]
      end

      local x, input_size_L
      local outputs = {}
      for L = 1,n do
         -- c,h from previos timesteps
         local prev_h = inputs[L*2+1]
         local prev_c = inputs[L*2]
         -- the input to this layer
         if L == 1 then
            x = nn.OneHot(input_size)(inputs[1])
            input_size_L = input_size
         else
            x = outputs[(L-1)*2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
         end
         -- evaluate the input sums at once for efficiency
         local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
         local h2h = nn.Linear(rnn_size, 4 * rnn_size):noBias()(prev_h):annotate{name='h2h_'..L}
         local all_input_sums = nn.CAddTable()({i2h, h2h})

         local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
         local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
         -- decode the gates
         local in_gate = nn.Sigmoid()(n1)
         local forget_gate = nn.Sigmoid()(n2)
         local out_gate = nn.Sigmoid()(n3)
         -- decode the write inputs
         local in_transform = nn.Tanh()(n4)
         -- perform the LSTM update
         local next_c           = nn.CAddTable()({
           nn.CMulTable()({forget_gate, prev_c}),
           nn.CMulTable()({in_gate,     in_transform})
         })
         -- gated cells form the output
         local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

         table.insert(outputs, next_c)
         table.insert(outputs, next_h)
      end

      -- set up the decoder
      local top_h = outputs[#outputs]
      if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
      local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
      local logsoft = nn.LogSoftMax()(proj)
      table.insert(outputs, logsoft)

      local lstm = nn.gModule(inputs, outputs):cuda()
      return lstm
   end

   -- the initial state of the cell/hidden states
   local init_state = {}
   for L=1,n_layer do
       local h_init = torch.zeros(batch_size, rnn_size):cuda()
       table.insert(init_state, h_init:clone())
       table.insert(init_state, h_init:clone())
   end

   local lstm1 = makeCharLSTM(input_size, rnn_size, n_layer)
   local crit1 = nn.ClassNLLCriterion()
   local protos = {rnn=lstm1,criterion=crit1}

   -- make a bunch of clones after flattening, as that reallocates memory
   local clones = {}
   for name,proto in pairs(protos) do
       clones[name] = model_utils.clone_many_times(proto, seq_len, not proto.parameters)
   end

   -- put the above things into one flattened parameters tensor
   local params, grad_params = model_utils.combine_all_parameters(lstm1)

   local init_state_global = clone_list(init_state)

   -- do fwd/bwd and return loss, grad_params
   local function trainCharrnn(x, y, fwdOnly)
      local rnn_state = {[0] = init_state_global}
      local predictions = {}           -- softmax outputs
      local loss = 0
      for t=1,seq_len do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        --loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
      end

      if not fwdOnly then
         --loss = loss / seq_len
         ------------------ backward pass -------------------
         -- initialize gradient at time t to be zeros (there's no influence from future)
         local drnn_state = {[seq_len] = clone_list(init_state, true)} -- true also zeros the clones
         for t=seq_len,1,-1 do
           -- backprop through loss, and softmax/linear
           --local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
           local doutput_t = y[t]
           table.insert(drnn_state[t], doutput_t)
           local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
           drnn_state[t-1] = {}
           for k,v in pairs(dlst) do
               if k > 1 then -- k == 1 is gradient on x, which we dont need
                   -- note we do k-1 because first item is dembeddings, and then follow the
                   -- derivatives of the state, starting at index 2. I know...
                   drnn_state[t-1][k-1] = v
               end
           end
         end
      end
      ------------------------ misc ----------------------
      -- transfer final state to initial state (BPTT)
      init_state_global = rnn_state[#rnn_state]
   end

   local charrnnsetuptime = a:time().real

   local a = torch.Timer()

   local function makeRnnLSTM(input_size, rnn_size, n)
      local seq = nn.Sequential()
         :add(nn.OneHot(input_size))

      local inputSize = input_size
      for L=1,n do
         seq:add(nn.RecLSTM(inputSize, rnn_size))
         inputSize = rnn_size
      end

      seq:add(nn.Linear(rnn_size, input_size))
      seq:add(nn.LogSoftMax())

      local lstm = nn.Sequencer(seq)

      lstm:cuda()

      return lstm
   end

   local lstm2 = makeRnnLSTM(input_size, rnn_size, n_layer, gpu)

   local function trainRnn(x, y, fwdOnly)
      local outputs = lstm2:forward(x)
      if not fwdOnly then
         local gradInputs = lstm2:backward(x, y)
      end
   end

   local rnnsetuptime = a:time().real

   -- char-rnn (nngraph)

   local a = torch.Timer()
   trainCharrnn(inputs, gradOutputs)
   cutorch.synchronize()
   charrnnsetuptime = charrnnsetuptime + a:time().real
   collectgarbage()

   local a = torch.Timer()
   for i=1,10 do
      trainCharrnn(inputs, gradOutputs)
   end
   cutorch.synchronize()
   local chartime = a:time().real

   -- rnn
   local a = torch.Timer()
   trainRnn(inputs, gradOutputs)
   cutorch.synchronize()
   rnnsetuptime = rnnsetuptime + a:time().real
   collectgarbage()
   print("Benchmark")
   print("setuptime : char, rnn, char/rnn", charrnnsetuptime, rnnsetuptime, charrnnsetuptime/rnnsetuptime)
   local a = torch.Timer()
   for i=1,10 do
      trainRnn(inputs, gradOutputs)
   end
   cutorch.synchronize()
   local rnntime = a:time().real
   print("runtime: char, rnn, char/rnn", chartime, rnntime, chartime/rnntime)

   -- on NVIDIA Titan Black :
   -- with FastLSTM.usenngraph = true :
   -- setuptime : char, rnn, char/rnn 1.5920469760895 2.4352579116821 0.65374881586558
   -- runtime: char, rnn, char/rnn    1.0614919662476 1.124755859375  0.94375322199913
end

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
   mlp:add(nn.Clamp(-1,1))
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
   lstms.luaseq = nn.SeqLSTM(inputsize, outputsize)
   lstms.luaseq.forceLua = true

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
      lstm.testtime = t:time().real/nloop
   end

   for i,name in ipairs{'fast','step','luarec','rec', 'luaseq', 'seq'} do
      print(name..' LSTM time: '..lstms[name].testtime..' seconds')
   end

   print("RecLSTM-C "..(lstms.luarec.testtime/lstms.rec.testtime)..' faster than RecLSTM-Lua')
   print("RecLSTM "..(lstms.fast.testtime/lstms.rec.testtime)..' faster than FastLSTM')
   print("SeqLSTM "..(lstms.rec.testtime/lstms.seq.testtime)..' faster than RecLSTM')
   print("SeqLSTM-C "..(lstms.luaseq.testtime/lstms.seq.testtime)..' faster than SeqLSTM-Lua')

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

function rnnbigtest.GRU()
   local seqlen, batchsize = 30, 32
   local inputsize, outputsize = 128, 128
   local nloop = 20

   local grus = {}
   grus.old = nn.Sequencer(nn.GRU(inputsize, outputsize))
   local stepmodule = nn.StepGRU(inputsize, outputsize)
   local recmodule = nn.Recurrence(stepmodule, outputsize, 1, seqlen)
   grus.step = nn.Sequencer(recmodule)
   grus.rec = nn.Sequencer(nn.RecGRU(inputsize, outputsize))
   local luarec = nn.RecGRU(inputsize, outputsize)
   luarec.modules[1].forceLua = true
   grus.luarec = nn.Sequencer(luarec)
   grus.seq = nn.SeqGRU(inputsize, outputsize)
   grus.luaseq = nn.SeqGRU(inputsize, outputsize)
   grus.luaseq.forceLua = true

   local input = torch.Tensor(seqlen, batchsize, inputsize)
   local gradOutput = torch.Tensor(seqlen, batchsize, outputsize)

   local t = torch.Timer()

   print("CPU test")

   for name, gru in pairs(grus) do
       -- warmup
      gru:remember('neither')
      gru:forward(input)
      gru:zeroGradParameters()
      gru:backward(input, gradOutput)
      -- main test
      t:reset()
      for i=1,nloop do
         gru:forward(input)
         gru:zeroGradParameters()
         gru:backward(input, gradOutput)
      end
      gru.testtime = t:time().real/nloop
   end

   for i,name in ipairs{'old','step','luarec','rec', 'luaseq', 'seq'} do
      print(name..' GRU time: '..grus[name].testtime..' seconds')
   end

   print("RecGRU-C "..(grus.luarec.testtime/grus.rec.testtime)..' faster than RecGRU-Lua')
   print("RecGRU "..(grus.old.testtime/grus.rec.testtime)..' faster than old GRU')
   print("SeqGRU "..(grus.rec.testtime/grus.seq.testtime)..' faster than RecGRU')
   print("SeqGRU-C "..(grus.luaseq.testtime/grus.seq.testtime)..' faster than SeqGRU-Lua')

   print("Memory test")

   for name, gru in pairs(grus) do
      gru.fullsize = #torch.serialize(gru)
      gru:clearState()
      gru.clearsize = #torch.serialize(gru)
   end

   for i,name in ipairs{'old','step','rec','seq'} do
      print(name..' GRU memory: '..grus[name].fullsize/(1024*1024)..':'..grus[name].clearsize/(1024*1024)..' MB')
   end
end


function rnn.bigtest(tests)
   mytester = torch.Tester()
   mytester:add(rnnbigtest)
   math.randomseed(os.time())
   mytester:run(tests)
end
