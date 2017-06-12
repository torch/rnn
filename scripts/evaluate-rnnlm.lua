require 'nngraph'
require 'rnn'
local dl = require 'dataload'


--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a RNNLM')
cmd:text('Options:')
cmd:option('--xplogpath', '', 'path to a previously saved xplog containing model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--device', 1, 'which GPU device to use')
cmd:option('--nsample', -1, 'sample this many words from the language model')
cmd:option('--temperature', 1, 'temperature of multinomial. Increase to sample wildly, reduce to be more deterministic.')
cmd:option('--dumpcsv', false, 'dump training and validation error to CSV file')
cmd:option('--bleu', false, 'BLEU evaluation')
cmd:option('--blueN', 4, 'N-grams for blue evaluation')
cmd:text()
local opt = cmd:parse(arg or {})

assert(opt.temperature > 0)

-- check that saved model exists
assert(paths.filep(opt.xplogpath), opt.xplogpath..' does not exist')

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

local xplog = torch.load(opt.xplogpath)
local lm = xplog.model
local criterion = xplog.criterion
local targetmodule = xplog.targetmodule

print("Hyper-parameters (xplog.opt):")
print(xplog.opt)

local trainerr = xplog.trainnceloss or xplog.trainppl
local validerr = xplog.valnceloss or xplog.valppl

print(string.format("Error (epoch=%d): training=%f; validation=%f", xplog.epoch, trainerr[#trainerr], validerr[#validerr]))


local function get_ngrams(sent, n, count)
   local ngrams = {}
   for beg = 1, #sent do
      for  last= beg, math.min(beg+n-1, #sent) do
         local ngram = table.concat(sent, ' ', beg, last)
         local len = last-beg+1 -- keep track of ngram length
         if not count then
            table.insert(ngrams, ngram)
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

local function get_ngram_prec(cand, ref, n)
   local results = {}
   for i = 1, n do
      results[i] = {0, 0}
   end
   local cand_ngrams = get_ngrams(cand, n, 1)
   local ref_ngrams = get_ngrams(ref, n, 1)
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

function get_bleu(cand, ref, n)
   n = n or 4
   local smooth = 1
   if type(cand) ~= 'table' then
      cand = cand:totable()
   end
   if type(ref) ~= 'table' then
      ref = ref:totable()
   end
   local res = get_ngram_prec(cand, ref, n)
   local brevPen = math.exp(1-math.max(1, #ref/#cand))
   local correct = 0
   local total = 0
   local bleu = 1
   for i = 1, n do
      if res[i][1] > 0 then
         if res[i][2] == 0 then
            smooth = smooth*0.5
            res[i][2] = smooth
         end
         local prec = res[i][2]/res[i][1]
         bleu = bleu * prec
      end
   end
   bleu = bleu^(1/n)
   return bleu*brevPen
end

if opt.dumpcsv then
   local csvfile = opt.xplogpath:match('([^/]+)[.]t7$')..'.csv'
   paths.mkdir('learningcurves')
   csvpath = paths.concat('learningcurves', csvfile)

   local file = io.open(csvpath, 'w')
   file:write("epoch,trainerr,validerr\n")
   for i=1,#trainerr do
      file:write(string.format('%d,%f,%f\n', i, trainerr[i], validerr[i]))
   end
   file:close()

   print("CSV file saved to "..csvpath)
   os.exit()
end

local trainset, validset, testset
if xplog.dataset == 'PennTreeBank' then
   print"Loading Penn Tree Bank test set"
   trainset, validset, testset = dl.loadPTB({50, 1, 1})
   assert(trainset.vocab['the'] == xplog.vocab['the'])
elseif xplog.dataset == 'GoogleBillionWords' then
   print"Loading Google Billion Words test set"
   trainset, validset, testset = dl.loadGBW({50,1,1}, 'train_tiny.th7')
else
   error"Unrecognized dataset"
end


for i,nce in ipairs(lm:findModules('nn.NCEModule')) do
   nce.normalized = true
   nce.logsoftmax = true
   if not opt.nce then
      print"Found NCEModule"
      criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
      if opt.cuda then criterion:cuda() end
      opt.nce = true
   end
end

print(lm)

lm:forget()
lm:evaluate()

if opt.nsample > 0 then
   if xplog.dataset == 'GoogleBillionWords' then
      local sampletext = {'<S>'}
      local prevword = trainset.vocab['<S>']
      assert(prevword)
      local inputs = torch.LongTensor(1,1) -- seqlen x batchsize
      local targets = opt.cuda and torch.CudaTensor(1) or torch.LongTensor(1) -- dummy tensor
      local buffer = torch.FloatTensor()
      for i=1,opt.nsample do
         inputs:fill(prevword)
         local output = lm:forward({inputs,{targets}})[1][1]
         buffer:resize(output:size()):copy(output)
         buffer:div(opt.temperature)
         buffer:exp()
         local sample = torch.multinomial(buffer, 1, true)
         local currentword = trainset.ivocab[sample[1]]
         table.insert(sampletext, currentword)
         if currentword == '</S>' then
            -- sentences were trained independently, so we explicitly tell it to start a new sentence
            lm:forget()
            prevword = trainset.vocab['<S>']
            table.insert(sampletext, '\n<S>')
         else
            prevword = sample[1]
         end
      end
      print(table.concat(sampletext, ' '))
   else
      local sampletext = {}
      local prevword = trainset.vocab['<eos>']
      assert(prevword)
      local inputs = torch.LongTensor(1,1) -- seqlen x batchsize
      if opt.cuda then inputs = inputs:cuda() end
      local buffer = torch.FloatTensor()
      for i=1,opt.nsample do
         inputs:fill(prevword)
         local output = lm:forward(inputs)[1][1]
         buffer:resize(output:size()):copy(output)
         buffer:div(opt.temperature)
         buffer:exp()
         local sample = torch.multinomial(buffer, 1, true)
         local currentword = trainset.ivocab[sample[1]]
         table.insert(sampletext, currentword)
         prevword = sample[1]
      end
      print(table.concat(sampletext, ' '))
   end
else
   local sumErr, count, sum_bleu, num_sent = 0, 0, 0, 0

   for i, inputs, targets in testset:subiter(xplog.opt.seqlen or 100) do
      inputs:apply(function(x)
         if x > 0 then
            count = count + 1
         end
      end)
      local targets = targetmodule:forward(targets)
      local inputs = opt.nce and {inputs, targets} or inputs
      local outputs = lm:forward(inputs)
      if opt.bleu then
         max_ind = torch.multinomial(torch.exp(outputs:view(targets:nElement(), -1)), 1):view(targets:size(1),targets:size(2))
         --max_ind = torch.multinomial(torch.exp(outputs:view(targets:size(1)*targets:size(2), -1)), 1):view(targets:size(1),targets:size(2))
            for batchIdx=1, targets:size(2) do
               sum_bleu = sum_bleu + get_bleu(max_ind:select(2, batchIdx),
                                              targets:select(2, batchIdx),
                                              opt.blueN)
               num_sent = num_sent + 1
            end
         end
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr + err
   end

   if count ~= testset:size() then
      local meanseqlen = testset:size()/(testset:size() - count)
      print("mean sequence length : "..meanseqlen)
   end

   local ppl = torch.exp(sumErr/count)
   print("Test PPL : "..ppl)
   if opt.bleu then
      print(" BLEU score : "..sum_bleu/num_sent)
   end
end

