local dl = require 'dataload'
require 'rnn'
require 'optim'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('-xplogpath', '', 'path to an xplog generated with examples/recurrent-visual-attention.lua')
cmd:option('-cuda', false, 'model was saved with cuda')
cmd:option('-evaltest', false, 'evaluate performance on test set')
cmd:option('-stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('-dataset', 'Mnist', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('-overwrite', false, 'overwrite checkpoint')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xplogpath), opt.xplogpath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xplog = torch.load(opt.xplogpath)
model = torch.type(xplog.model) == 'nn.Serial' and xplog.model.modules[1] or xplog.model

print("Last evaluation of validation set")
print(xplog.validcm[#xplog.validcm])

--[[
if opt.dataset == 'TranslatedMnist' then
   ds = torch.checkpoint(
      paths.concat(dp.DATA_DIR, 'checkpoint/dp.TranslatedMnist_test.t7'),
      function()
         local ds = dp[opt.dataset]{load_all=false}
         ds:loadTest()
         return ds
         end,
      opt.overwrite
   )
else
   ds = dp[opt.dataset]()
end
--]]
assert(opt.dataset == 'Mnist')
trainset, validset, testset = dl.loadMNIST()

ra = model:findModules('nn.RecurrentAttention')[1]
sg = model:findModules('nn.SpatialGlimpse')[1]

-- stochastic or deterministic
for i=1,#ra.actions do
   local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
   rn.stochastic = opt.stochastic
end

local testcm = optim.ConfusionMatrix(10)
if opt.evaltest then
   model:evaluate()
   for i, input, target in testset:subiter(opt.batchsize) do
      target = xplog.targetmodule:forward(target)
      local output = model:forward(input)
      testcm:batchAdd(output[1], target)
   end

   print((opt.stochastic and "Stochastic" or "Deterministic") .. " evaluation of test set :")
   print(testcm)
end


input = testset.inputs:narrow(1,1,10)
model:training() -- otherwise the rnn doesn't save intermediate time-step states
if not opt.stochastic then
   for i=1,#ra.actions do
      local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
      rn.stdev = 0 -- deterministic
   end
end
output = model:forward(input)

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

    local max = img:max()

    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

locations = ra.actions
glimpses = {}
patches = {}

params = nil
for i=1,input:size(1) do
   local img = input[i]
   for j,location in ipairs(locations) do
      local glimpse = glimpses[j] or {}
      glimpses[j] = glimpse
      local patch = patches[j] or {}
      patches[j] = patch

      local xy = location[i]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- (1,1), (input:size(3), input:size(4))
      x, y = x*(input:size(3)-1)+1, y*(input:size(4)-1)+1

      local gimg = img:clone()
      for d=1,sg.depth do
         local size = sg.height*(sg.scale^(d-1))
         local bbox = {y-size/2, x-size/2, size, size}
         drawBox(gimg, bbox, 1)
      end
      glimpse[i] = gimg

      local sg_ = ra.rnn.sharedClones[j]:findModules('nn.SpatialGlimpse')[1]
      patch[i] = image.scale(img:clone():float(), sg_.output[i]:narrow(1,1,1):float())

      collectgarbage()
   end
end

paths.mkdir('glimpse')
for j,glimpse in ipairs(glimpses) do
   local g = image.toDisplayTensor{input=glimpse,nrow=10,padding=3}
   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
   image.save("glimpse/glimpse"..j..".png", g)
   image.save("glimpse/patch"..j..".png", p)
end


