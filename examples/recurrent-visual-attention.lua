local dl = require 'dataload'
require 'rnn'
require 'optim'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('-startlr', 0.01, 'learning rate at t=0')
cmd:option('-minlr', 0.00001, 'minimum learning rate')
cmd:option('-saturate', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-maxnormout', -1, 'max norm each layers output neuron weights')
cmd:option('-cutoff', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('-batchsize', 20, 'number of examples per batch')
cmd:option('-cuda', false, 'use CUDA')
cmd:option('-device', 1, 'sets the device (GPU) to use')
cmd:option('-maxepoch', 2000, 'maximum number of epochs to run')
cmd:option('-earlystop', 200, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('-transfer', 'ReLU', 'activation function')
cmd:option('-uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('-progress', false, 'print progress bar')
cmd:option('-silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('-rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('-unitPixels', 13, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('-locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('-stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
cmd:option('-glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('-glimpsePatchSize', 8, 'size of glimpse patch at highest res (height = width)')
cmd:option('-glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('-glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('-locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('-imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
cmd:option('-seqlen', 7, 'back-propagate through time (BPTT) for seqlen time-steps')
cmd:option('-hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('-lstm', false, 'use LSTM instead of linear layer')

--[[ data ]]--
cmd:option('-dataset', 'Mnist', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('-trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('-validsize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('-noTest', false, 'dont propagate through the test set')
cmd:option('-overwrite', false, 'overwrite checkpoint')
cmd:option('-savepath', paths.concat(dl.SAVE_PATH, 'rmva'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('-id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})
opt.version = 13
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id
if not opt.silent then
   table.print(opt)
end

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

--[[data]]--
--[[if opt.dataset == 'TranslatedMnist' then
   ds = torch.checkpoint(
      paths.concat(dp.DATA_DIR, 'checkpoint/dp.TranslatedMnist.t7'),
      function() return dp[opt.dataset]() end,
      opt.overwrite
   )
else
   ds = dp[opt.dataset]()
end--]]

assert(opt.dataset == 'Mnist')
trainset, validset, testset = dl.loadMNIST()

--[[Model]]--

-- glimpse network (rnn input layer)
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())

glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float())
glimpseSensor:add(nn.Collapse(3))
glimpseSensor:add(nn.Linear(trainset:isize()[1]*(opt.glimpsePatchSize^2)*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())

-- RNN layer
if opt.lstm then
   glimpse:add(nn.RecLSTM(opt.imageHiddenSize, opt.hiddenSize))
else
   glimpse:add(nn.LinearRNN(opt.imageHiddenSize, opt.hiddenSize, nn[opt.transfer]()))
end

imageSize = trainset:isize()
assert(imageSize[2] == imageSize[3])

-- actions (locator)
locator = nn.Sequential()
locator:add(nn.Linear(opt.hiddenSize, 2))
locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(opt.unitPixels*2/imageSize[2]))

attention = nn.RecurrentAttention(glimpse, locator, opt.seqlen, {opt.hiddenSize})

-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert())
agent:add(attention)

-- classifier :
agent:add(nn.SelectTable(-1))
agent:add(nn.Linear(opt.hiddenSize, #testset.classes))
agent:add(nn.LogSoftMax())

-- add the baseline reward predictor
seq = nn.Sequential()
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
agent:add(concat2)

if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

print("Recurrent visual attention model:")
print(agent)

-- [[Criterion]]

criterion = nn.ParallelCriterion(true)
   :add(nn.ClassNLLCriterion()) -- BACKPROP
   :add(nn.VRClassReward(agent, opt.rewardScale)) -- REINFORCE

targetmodule = nn.Convert()


--[[ CUDA ]]--

if opt.cuda then
   agent:cuda()
   criterion:cuda()
   targetmodule:cuda()
else
   agent:float()
   criterion:float()
   targetmodule:float()
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
-- will only serialize params
xplog.model = agent:sharedClone()
xplog.criterion = criterion
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.traincm = {}
xplog.validcm = {}
-- will be used for early-stopping
xplog.minvaliderr = 99999999
xplog.epoch = 0

--[[ training loop ]]--

local ntrial = 0
paths.mkdir(opt.savepath)

local epoch = 1
opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")

   local traincm = optim.ConfusionMatrix(10)

   -- 1. training

   local a = torch.Timer()
   agent:training()
   for i, input, target in trainset:sampleiter(opt.batchsize, opt.trainsize) do
      target = targetmodule:forward(target)
      -- forward
      local output = agent:forward(input)
      local err = criterion:forward(output, target)
      traincm:batchAdd(output[1], target)

      -- backward
      local gradOutput = criterion:backward(output, target)
      agent:zeroGradParameters()
      agent:backward(input, gradOutput)

      -- update
      if opt.cutoff > 0 then
         local norm = agent:gradParamClip(opt.cutoff) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      agent:updateGradParameters(opt.momentum) -- affects gradParams
      agent:updateParameters(opt.lr) -- affects params
      agent:maxParamNorm(opt.maxnormout) -- affects params

      if opt.progress then
         xlua.progress(i, opt.trainsize)
      end

      if i % 1000 == 0 then
         collectgarbage()
      end

   end

   -- learning rate decay
   opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
   opt.lr = math.max(opt.minlr, opt.lr)

   if not opt.silent then
      print("learning rate", opt.lr)
      if opt.meanNorm then
         print("mean gradParam norm", opt.meanNorm)
      end
   end

   if cutorch then cutorch.synchronize() end
   local speed = a:time().real/opt.trainsize
   print(string.format("Speed : %f sec/batch ", speed))

   traincm:updateValids()
   print("Training error : "..((1 - traincm.totalValid)*100).."%")

   xplog.traincm[epoch] = traincm

   -- 2. cross-validation

   agent:evaluate()
   local validcm = optim.ConfusionMatrix(10)
   for i, input, target in validset:subiter(opt.batchsize, opt.validsize) do
      target = targetmodule:forward(target)
      local output = agent:forward(input)
      validcm:batchAdd(output[1], target)
   end

   validcm:updateValids()
   local validerr = 1 - validcm.totalValid
   print("Validation error : "..(validerr*100).."%")

   xplog.validcm[epoch] = validcm
   ntrial = ntrial + 1

   -- early-stopping
   if validerr < xplog.minvaliderr then
      -- save best version of model
      xplog.minvaliderr = validerr
      xplog.epoch = epoch
      local filename = paths.concat(opt.savepath, opt.id..'.t7')
      print("Found new minima. Saving to "..filename)
      torch.save(filename, xplog)
      ntrial = 0
   elseif ntrial >= opt.earlystop then
      print("No new minima found after "..ntrial.." epochs.")
      print("Stopping experiment.")
      break
   end


   collectgarbage()
   epoch = epoch + 1
end
print("Evaluate model using : ")
print("th scripts/evaluate-rva.lua -xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and ' -cuda' or '')..' -evaltest')
