local _ = require 'moses'
local rnntest = torch.TestSuite()
local precision = 1e-5
local mytester
local benchmark = false

function rnntest.RecLSTM_main()
   local batchSize = math.random(1,2)
   local inputSize = math.random(3,4)
   local outputSize = math.random(5,6)
   local nStep = 3
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, inputSize)
      if step == nStep then
         -- for the sake of keeping this unit test simple,
         gradOutput[step] = torch.randn(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput[step] = torch.zeros(batchSize, outputSize)
      end
   end
   local lstm = nn.RecLSTM(inputSize, outputSize)

   -- we will use this to build an LSTM step by step (with shared params)
   local lstmStep = lstm.modules[1]:clone()

   -- forward/backward through LSTM
   local output = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output[step] = lstm:forward(input[step])
      assert(torch.isTensor(input[step]))
   end

   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = lstm:backward(input[step], gradOutput[step], 1)
   end

   local gradInput = gradInputs[1]

   local mlp2 -- this one will simulate seqlen = nStep
   local inputs
   for step=1,nStep do
      -- iteratively build an LSTM out of non-recurrent components
      local lstm = lstmStep:clone()
      lstm:share(lstmStep, 'weight', 'gradWeight', 'bias', 'gradBias')
      if step == 1 then
         mlp2 = lstm
      else
         local rnn = nn.Sequential()
         local para = nn.ParallelTable()
         para:add(nn.Identity()):add(mlp2)
         rnn:add(para)
         rnn:add(nn.FlattenTable())
         rnn:add(lstm)
         mlp2 = rnn
      end

      -- prepare inputs for mlp2
      if inputs then
         inputs = {input[step], inputs}
      else
         inputs = {input[step], torch.zeros(batchSize, outputSize), torch.zeros(batchSize, outputSize)}
      end
   end
   mlp2:add(nn.SelectTable(1)) --just output the output (not cell)
   local output2 = mlp2:forward(inputs)

   mlp2:zeroGradParameters()
   local gradInput2 = mlp2:backward(inputs, gradOutput[nStep], 1) --/nStep)
   mytester:assertTensorEq(gradInput2[2][2][1], gradInput, 0.00001, "LSTM gradInput error")
   mytester:assertTensorEq(output[nStep], output2, 0.00001, "LSTM output error")

   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = lstmStep:parameters()
   mytester:assert(#params == #params2, "LSTM parameters error "..#params.." ~= "..#params2)
   for i, gradParam in ipairs(gradParams) do
      local gradParam2 = gradParams2[i]
      mytester:assertTensorEq(gradParam, gradParam2, 0.000001,
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam2))
   end

   gradParams = nn.utils.recursiveCopy(nil, gradParams)
   gradInput = gradInput:clone()
   mytester:assert(lstm.zeroOutput:sum() == 0, "zeroOutput error")
   mytester:assert(lstm.zeroCell:sum() == 0, "zeroCell error")
   lstm:forget()
   output = nn.utils.recursiveCopy(nil, output)
   local output3 = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output3[step] = lstm:forward(input[step])
   end

   local gradInputs3 = {}
   for step=nStep,1,-1 do
      gradInputs3[step] = lstm:updateGradInput(input[step], gradOutput[step])
      lstm:accGradParameters(input[step], gradOutput[step], 1)
   end
   local gradInput3 = gradInputs[1]

   mytester:assert(#output == #output3, "LSTM output size error")
   for i,output in ipairs(output) do
      mytester:assertTensorEq(output, output3[i], 0.00001, "LSTM forget (updateOutput) error "..i)
   end

   mytester:assertTensorEq(gradInput, gradInput3, 0.00001, "LSTM updateGradInput error")

   local params3, gradParams3 = lstm:parameters()
   mytester:assert(#params == #params3, "LSTM parameters error "..#params.." ~= "..#params3)
   for i, gradParam in ipairs(gradParams) do
      local gradParam3 = gradParams3[i]
      mytester:assertTensorEq(gradParam, gradParam3, 0.000001,
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam3))
   end
end

function rnntest.RecurrentAttention()
   if not pcall(function() require 'nnx' end) then return end

   local opt = {
      glimpseDepth = 3,
      glimpseHiddenSize = 20,
      glimpsePatchSize = 8,
      locatorHiddenSize = 20,
      imageHiddenSize = 20,
      hiddenSize = 20,
      seqlen = 5,
      locatorStd = 0.1,
      inputSize = 28,
      nClass = 10,
      batchSize = 4
   }

   -- glimpse network (rnn input layer)
   local locationSensor = nn.Sequential()
   locationSensor:add(nn.SelectTable(2))
   locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
   locationSensor:add(nn.ReLU())

   local glimpseSensor = nn.Sequential()
   glimpseSensor:add(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale))
   glimpseSensor:add(nn.Collapse(3))
   glimpseSensor:add(nn.Linear(1*(opt.glimpsePatchSize^2)*opt.glimpseDepth, opt.glimpseHiddenSize))
   glimpseSensor:add(nn.ReLU())

   local glimpse = nn.Sequential()
   --glimpse:add(nn.PrintSize("preglimpse"))
   glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
   glimpse:add(nn.JoinTable(1,1))
   glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
   glimpse:add(nn.ReLU())
   glimpse:add(nn.LinearRNN(opt.imageHiddenSize, opt.hiddenSize))

   -- recurrent neural network
   local rnn = nn.Recursor(glimpse)

   -- output layer (actions)
   local locator = nn.Sequential()
   locator:add(nn.Linear(opt.hiddenSize, 2))
   locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
   local rn = nn.ReinforceNormal(2*opt.locatorStd)
   rn:evaluate() -- so we can match the output from sg to sg2 (i.e deterministic)
   locator:add(rn) -- sample from normal, uses REINFORCE learning rule
   locator:add(nn.HardTanh()) -- bounds sample between -1 and 1

   -- model is a reinforcement learning agent
   local rva = nn.RecurrentAttention(rnn:clone(), locator:clone(), opt.seqlen, {opt.hiddenSize})

   local input = torch.randn(opt.batchSize,1,opt.inputSize,opt.inputSize)
   local gradOutput = {}
   for step=1,opt.seqlen do
      table.insert(gradOutput, torch.randn(opt.batchSize, opt.hiddenSize))
   end

   rva:zeroGradParameters()

   local output = rva:forward(input)

   mytester:assert(#output == opt.seqlen, "RecurrentAttention #output err")

   local reward = torch.randn(opt.batchSize)
   rva:reinforce(reward)
   local gradInput = rva:backward(input, gradOutput)

   mytester:assert(gradInput:isSameSizeAs(input))

   rva:updateParameters(1)
end

function rnntest.Sequencer_main()
   local batchSize = 4
   local inputSize = 3
   local outputSize = 7
   local nStep = 5

   local stepmodule = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(inputSize, outputSize))
         :add(nn.Euclidean(outputSize, outputSize)))
      :add(nn.CAddTable())
      :add(nn.Sigmoid())

   local rnn = nn.Recurrence(stepmodule, outputSize, 1)
   rnn:zeroGradParameters()
   local rnn2 = rnn:clone()

   local inputs, outputs, gradOutputs = {}, {}, {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
      outputs[step] = rnn:forward(inputs[step]):clone()
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end

   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end

   local gradOutput1 = gradOutputs[1]:clone()
   local rnn3 = nn.Sequencer(rnn2)
   local outputs3 = rnn3:forward(inputs)
   mytester:assert(#outputs3 == #outputs, "Sequencer output size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer output "..step)
   end
   local gradInputs3 = rnn3:backward(inputs, gradOutputs)

   mytester:assert(#gradInputs3 == #gradInputs, "Sequencer gradInputs size err")
   mytester:assert(gradInputs3[1]:nElement() ~= 0)

   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(gradInputs3[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[1], gradOutput1, 0.00001, "Sequencer rnn gradOutput modified error")

   local nStep7 = torch.Tensor{5,4,5,3,7,3,3,3}
   local function testRemember(rnn)
      rnn:zeroGradParameters()
      -- test remember for training mode (with variable length)
      local rnn7 = rnn:clone()
      rnn7:zeroGradParameters()
      local rnn8 = rnn7:clone()
      local rnn9 = rnn7:clone()
      local rnn10 = nn.Recursor(rnn7:clone())

      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local outputs7, gradOutputs7, gradInputs7 = {}, {}, {}
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            gradOutputs7[step] = torch.randn(batchSize, outputSize)
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step]):clone()
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end

      -- nn.Recursor

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn10:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn10:forget()

      local step = 1
      local outputs10, gradOutputs10, gradInputs10 = {}, {}, {}
      for i=1,nStep7:size(1) do
         local start = step
         local nStep = 0
         for j=1,nStep7[i] do
            outputs10[step] = rnn10:forward(inputs7[step]):clone()
            step = step + 1
            nStep = nStep + 1
         end
         rnn10:maxBPTTstep(nStep7[i])
         local nStep2 = 0
         for s=step-1,start,-1 do
            gradInputs10[s] = rnn10:backward(inputs7[s], gradOutputs7[s]):clone()
            nStep2 = nStep2 + 1
         end
         assert(nStep == nStep2)
         rnn10:updateParameters(1)
         rnn10:zeroGradParameters()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs10[step], gradInputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs10[step], outputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      -- nn.Sequencer

      local seq = nn.Sequencer(rnn8)
      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         gradInputs8[i] = _.map(seq:backward(inputs8, gradOutputs8), function(k,v) return v:clone() end)
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end

      -- test in evaluation mode with remember and variable seqlen
      local rnn7 = rnn:clone() -- a fresh copy (no hidden states)
      local params7 = rnn7:parameters()
      local params9 = rnn9:parameters() -- not a fresh copy
      for i,param in ipairs(rnn8:parameters()) do
         params7[i]:copy(param)
         params9[i]:copy(param)
      end

      rnn7:evaluate()
      rnn9:evaluate()
      rnn9:forget()

      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local outputs7 = {}
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            step = step + 1
         end
      end

      seq:remember('both')
      local outputs8 = {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:evaluate()
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable eval forward err "..i.." "..j)
            step = step + 1
         end
      end

      -- test remember for training mode (with variable length) (from evaluation to training)

      rnn7:forget()
      rnn9:forget()

      rnn7:training()
      rnn9:training()

      rnn7:zeroGradParameters()
      seq:zeroGradParameters()
      rnn9:zeroGradParameters()

      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local outputs7, gradOutputs7, gradInputs7 = {}, {}, {}
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            gradOutputs7[step] = torch.randn(batchSize, outputSize)
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step]):clone()
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end

      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:training()
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         gradInputs8[i] = _.map(seq:backward(inputs8, gradOutputs8), function(k,v) return v:clone() end)
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end
   end
   local linearRNN = nn.LinearRNN(outputSize, outputSize)
   linearRNN:maxBPTTstep(nStep7:max())
   testRemember(linearRNN)
   local recLSTM = nn.RecLSTM(outputSize, outputSize)
   recLSTM:maxBPTTstep(nStep7:max())
   testRemember(recLSTM)

   -- test in evaluation mode
   rnn3:evaluate()
   local outputs4 = rnn3:forward(inputs)
   local outputs4_ = _.map(outputs4, function(k,v) return v:clone() end)
   mytester:assert(#outputs4 == #outputs, "Sequencer evaluate output size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs4[step], output, 0.00001, "Sequencer evaluate output "..step)
   end
   local inputs5 = _.clone(inputs)
   table.remove(inputs5, nStep) -- remove last input
   local outputs5 = rnn3:forward(inputs5)
   mytester:assert(#outputs5 == #outputs - 1, "Sequencer evaluate -1 output size err")
   for step,output in ipairs(outputs5) do
      mytester:assertTensorEq(outputs[step], output, 0.00001, "Sequencer evaluate -1 output "..step)
   end

   -- test evaluation with remember
   rnn3:remember()
   rnn3:evaluate()
   rnn3:forget()
   local inputsA, inputsB = {inputs[1],inputs[2],inputs[3]}, {inputs[4],inputs[5]}
   local outputsA = _.map(rnn3:forward(inputsA), function(k,v) return v:clone() end)
   local outputsB = rnn3:forward(inputsB)
   mytester:assert(#outputsA == 3, "Sequencer evaluate-remember output size err A")
   mytester:assert(#outputsB == 2, "Sequencer evaluate-remember output size err B")
   local outputsAB = {unpack(outputsA)}
   outputsAB[4], outputsAB[5] = unpack(outputsB)
   for step,output in ipairs(outputs4_) do
      mytester:assertTensorEq(outputsAB[step], output, 0.00001, "Sequencer evaluate-remember output "..step)
   end

   -- test with non-recurrent module
   local inputSize = 10
   local inputs = {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
   end
   local linear = nn.Euclidean(inputSize, outputSize)
   local seq, outputs, gradInputs
   for k=1,3 do
      outputs, gradInputs = {}, {}
      linear:zeroGradParameters()
      local clone = linear:clone()
      for step=1,nStep do
         outputs[step] = linear:forward(inputs[step]):clone()
         gradInputs[step] = linear:backward(inputs[step], gradOutputs[step]):clone()
      end

      seq = nn.Sequencer(clone)
      local outputs2 = seq:forward(inputs)
      local gradInputs2 = seq:backward(inputs, gradOutputs)

      mytester:assert(#outputs2 == #outputs, "Sequencer output size err")
      mytester:assert(#gradInputs2 == #gradInputs, "Sequencer gradInputs size err")
      for step,output in ipairs(outputs) do
         mytester:assertTensorEq(outputs2[step], output, 0.00001, "Sequencer output "..step)
         mytester:assertTensorEq(gradInputs2[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
      end
   end

   local inputs3, gradOutputs3 = {}, {}
   for i=1,#inputs do
      inputs3[i] = inputs[i]:float()
      gradOutputs3[i] = gradOutputs[i]:float()
   end
   local seq3 = seq:float()
   local outputs3 = seq:forward(inputs3)
   local gradInputs3 = seq:backward(inputs3, gradOutputs3)

   -- test for zeroGradParameters
   local seq = nn.Sequencer(nn.Linear(inputSize,outputSize))
   seq:zeroGradParameters()
   seq:forward(inputs)
   seq:backward(inputs, gradOutputs)
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() ~= 0, "Sequencer:backward err "..i)
   end
   local param, gradParam = seq:getParameters()
   seq:zeroGradParameters()
   mytester:assert(gradParam:sum() == 0, "Sequencer:getParameters err")
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() == 0, "Sequencer:zeroGradParameters err "..i)
   end

   -- test with LSTM
   local outputSize = inputSize
   local lstm = nn.LSTM(inputSize, outputSize, nil, false)
   lstm:zeroGradParameters()
   local lstm2 = lstm:clone()

   local inputs, outputs, gradOutputs = {}, {}, {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end
   local gradOutput1 = gradOutputs[2]:clone()
   for step=1,nStep do
      outputs[step] = lstm:forward(inputs[step])
   end

   local gradInputs72 = {}
   for step=nStep,1,-1 do
      gradInputs72[step] = lstm:backward(inputs[step], gradOutputs[step])
   end

   local lstm3 = nn.Sequencer(lstm2)
   lstm3:zeroGradParameters()
   local outputs3 = lstm3:forward(inputs)
   local gradInputs3 = lstm3:backward(inputs, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Sequencer LSTM output size err")
   mytester:assert(#gradInputs3 == #gradInputs72, "Sequencer LSTM gradInputs size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer LSTM output "..step)
      mytester:assertTensorEq(gradInputs3[step], gradInputs72[step], 0.00001, "Sequencer LSTM gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[2], gradOutput1, 0.00001, "Sequencer lstm gradOutput modified error")

   -- test remember modes : 'both', 'eval' for training(), evaluate(), training()
   local lstm = nn.LSTM(5,5)
   local seq = nn.Sequencer(lstm)
   local inputTrain = {torch.randn(5), torch.randn(5), torch.randn(5)}
   local inputEval = {torch.randn(5)}

   -- this shouldn't fail
   local modes = {'both', 'eval'}
   for i, mode in ipairs(modes) do
     seq:remember(mode)

     -- do one epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)

     -- evaluate
     seq:evaluate()
     seq:forward(inputEval)

     -- do another epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)
   end
end

function rnntest.Sequencer_tensor()
   -- test Sequencer where input/gradOutput are tensors instead of tables
   local batchSize = 4
   local inputSize = 3
   local outputSize = 7
   local nStep = 5

   -- test with recurrent module
   local stepmodule = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(inputSize, outputSize))
         :add(nn.Euclidean(outputSize, outputSize)))
      :add(nn.CAddTable())
      :add(nn.Sigmoid())

   local rnn = nn.Recurrence(stepmodule, outputSize, 1)
   rnn:zeroGradParameters()
   local rnn2 = rnn:clone()

   local outputs = torch.Tensor(nStep, batchSize, outputSize)
   local inputs = torch.randn(nStep, batchSize, inputSize)
   local gradOutputs = torch.randn(nStep, batchSize, outputSize)
   for step=1,nStep do
      outputs[step] = rnn:forward(inputs[step]):clone()
   end

   local gradInputs = torch.Tensor(nStep, batchSize, inputSize)
   for step=nStep,1,-1 do
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end

   local gradOutput1 = gradOutputs[1]:clone()
   local rnn3 = nn.Sequencer(rnn2)
   local outputs3 = rnn3:forward(inputs)
   mytester:assert(outputs3:size(1) == outputs:size(1), "Sequencer output size err")
   for step=1,nStep do
      mytester:assertTensorEq(outputs3[step], outputs[step], 0.00001, "Sequencer output "..step)
   end
   local gradInputs3 = rnn3:backward(inputs, gradOutputs)

   mytester:assert(gradInputs3:size(1) == gradInputs:size(1), "Sequencer gradInputs size err")
   mytester:assert(gradInputs3[1]:nElement() ~= 0)

   for step=1,nStep do
      mytester:assertTensorEq(gradInputs3[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[1], gradOutput1, 0.00001, "Sequencer rnn gradOutput modified error")

   local nStep7 = torch.Tensor{5,4,5,3,7,3,3,3}
   local function testRemember(rnn)
      rnn:zeroGradParameters()
      -- test remember for training mode (with variable length)
      local rnn7 = rnn:clone()
      rnn7:zeroGradParameters()
      local rnn8 = rnn7:clone()
      local rnn9 = rnn7:clone()
      local rnn10 = nn.Recursor(rnn7:clone())

      local inputs7 = torch.randn(nStep7:sum(), batchSize, outputSize)
      local outputs9 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      for step=1,nStep7:sum() do
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local gradOutputs7 = torch.randn(nStep7:sum(), batchSize, outputSize)
      local outputs7 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      local gradInputs7 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step])
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step])
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end

      -- nn.Recursor

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn10:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn10:forget()

      local step = 1
      local outputs10, gradOutputs10, gradInputs10 = {}, {}, {}
      for i=1,nStep7:size(1) do
         local start = step
         local nStep = 0
         for j=1,nStep7[i] do
            outputs10[step] = rnn10:forward(inputs7[step]):clone()
            step = step + 1
            nStep = nStep + 1
         end
         rnn10:maxBPTTstep(nStep7[i])
         local nStep2 = 0
         for s=step-1,start,-1 do
            gradInputs10[s] = rnn10:backward(inputs7[s], gradOutputs7[s]):clone()
            nStep2 = nStep2 + 1
         end
         assert(nStep == nStep2)
         rnn10:updateParameters(1)
         rnn10:zeroGradParameters()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs10[step], gradInputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs10[step], outputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      -- nn.Sequencer

      local seq = nn.Sequencer(rnn8)
      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         local inputs8 = inputs7:sub(step,step+nStep7[i]-1)
         local gradOutputs8 = gradOutputs7:sub(step,step+nStep7[i]-1)
         outputs8[i] = seq:forward(inputs8):clone()
         gradInputs8[i] = seq:backward(inputs8, gradOutputs8):clone()
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end

      -- test in evaluation mode with remember and variable seqlen
      local rnn7 = rnn:clone() -- a fresh copy (no hidden states)
      local params7 = rnn7:parameters()
      local params9 = rnn9:parameters() -- not a fresh copy
      for i,param in ipairs(rnn8:parameters()) do
         params7[i]:copy(param)
         params9[i]:copy(param)
      end

      rnn7:evaluate()
      rnn9:evaluate()
      rnn9:forget()

      local inputs7 = torch.randn(nStep7:sum(), batchSize, outputSize)
      local outputs9 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      for step=1,nStep7:sum() do
         outputs9[step] = rnn9:forward(inputs7[step])
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local outputs7 = {}
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            step = step + 1
         end
      end

      seq:remember('both')
      local outputs8 = {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:evaluate()
         local inputs8 = inputs7:sub(step,step+nStep7[i]-1)
         local gradOutputs8 = gradOutputs7:sub(step,step+nStep7[i]-1)
         outputs8[i] = seq:forward(inputs8):clone()
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable eval forward err "..i.." "..j)
            step = step + 1
         end
      end

      -- test remember for training mode (with variable length) (from evaluation to training)

      rnn7:forget()
      rnn9:forget()

      rnn7:training()
      rnn9:training()

      rnn7:zeroGradParameters()
      seq:zeroGradParameters()
      rnn9:zeroGradParameters()

      local inputs7 = torch.randn(nStep7:sum(), batchSize, outputSize)
      local outputs9 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end

      rnn7:forget()

      local step = 1
      local outputs7 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      local gradOutputs7 = torch.randn(nStep7:sum(), batchSize, outputSize)
      local gradInputs7 = torch.Tensor(nStep7:sum(), batchSize, outputSize)
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step])
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step])
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end

      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:training()
         local inputs8 = inputs7:sub(step,step+nStep7[i]-1)
         local gradOutputs8 = gradOutputs7:sub(step,step+nStep7[i]-1)
         outputs8[i] = seq:forward(inputs8):clone()
         gradInputs8[i] = seq:backward(inputs8, gradOutputs8):clone()
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end

      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end

      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end
   end

   local linearRNN = nn.LinearRNN(outputSize, outputSize)
   linearRNN:maxBPTTstep(nStep7:max())
   testRemember(linearRNN)
   local recLSTM = nn.RecLSTM(outputSize, outputSize)
   recLSTM:maxBPTTstep(nStep7:max())
   testRemember(recLSTM)

   -- test in evaluation mode
   rnn3:evaluate()
   local outputs4 = rnn3:forward(inputs)
   local outputs4_ = outputs4:clone()
   mytester:assert(outputs4:size(1) == outputs:size(1), "Sequencer evaluate output size err")
   for step=1,nStep do
      mytester:assertTensorEq(outputs4[step], outputs[step], 0.00001, "Sequencer evaluate output "..step)
   end
   local inputs5 = inputs:sub(1,nStep-1) -- remove last input
   local outputs5 = rnn3:forward(inputs5)
   mytester:assert(outputs5:size(1) == outputs:size(1) - 1, "Sequencer evaluate -1 output size err")
   for step=1,nStep-1 do
      mytester:assertTensorEq(outputs[step], outputs5[step], 0.00001, "Sequencer evaluate -1 output "..step)
   end

   -- test evaluation with remember
   rnn3:remember()
   rnn3:evaluate()
   rnn3:forget()
   local inputsA, inputsB = inputs:sub(1,3), inputs:sub(4,5)
   local outputsA = rnn3:forward(inputsA):clone()
   local outputsB = rnn3:forward(inputsB)
   mytester:assert(outputsA:size(1) == 3, "Sequencer evaluate-remember output size err A")
   mytester:assert(outputsB:size(1) == 2, "Sequencer evaluate-remember output size err B")
   local outputsAB = {outputsA[1], outputsA[2], outputsA[3], outputsB[1], outputsB[2]}
   for step=1,5 do
      mytester:assertTensorEq(outputsAB[step], outputs4_[step], 0.00001, "Sequencer evaluate-remember output "..step)
   end

   -- test with non-recurrent module
   local inputSize = 10
   local inputs = torch.randn(nStep, batchSize, inputSize)
   local linear = nn.Euclidean(inputSize, outputSize)
   local seq, outputs, gradInputs
   for k=1,3 do
      outputs, gradInputs = {}, {}
      linear:zeroGradParameters()
      local clone = linear:clone()
      for step=1,nStep do
         outputs[step] = linear:forward(inputs[step]):clone()
         gradInputs[step] = linear:backward(inputs[step], gradOutputs[step]):clone()
      end

      seq = nn.Sequencer(clone)
      local outputs2 = seq:forward(inputs)
      local gradInputs2 = seq:backward(inputs, gradOutputs)

      mytester:assert(outputs2:size(1) == #outputs, "Sequencer output size err")
      mytester:assert(gradInputs2:size(1) == #gradInputs, "Sequencer gradInputs size err")
      for step,output in ipairs(outputs) do
         mytester:assertTensorEq(outputs2[step], output, 0.00001, "Sequencer output "..step)
         mytester:assertTensorEq(gradInputs2[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
      end
   end

   local inputs3 = inputs:float()
   local gradOutputs3 = gradOutputs:float()
   local seq3 = seq:float()
   local outputs3 = seq:forward(inputs3)
   local gradInputs3 = seq:backward(inputs3, gradOutputs3)

   -- test for zeroGradParameters
   local seq = nn.Sequencer(nn.Linear(inputSize,outputSize))
   seq:zeroGradParameters()
   seq:forward(inputs)
   seq:backward(inputs, gradOutputs)
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() ~= 0, "Sequencer:backward err "..i)
   end
   local param, gradParam = seq:getParameters()
   seq:zeroGradParameters()
   mytester:assert(gradParam:sum() == 0, "Sequencer:getParameters err")
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() == 0, "Sequencer:zeroGradParameters err "..i)
   end

   -- test with LSTM
   local outputSize = inputSize
   local lstm = nn.LSTM(inputSize, outputSize, nil, false)
   lstm:zeroGradParameters()
   local lstm2 = lstm:clone()

   local inputs = torch.randn(nStep, batchSize, inputSize)
   local outputs = torch.Tensor(nStep, batchSize, outputSize)
   local gradOutputs = torch.randn(nStep, batchSize, outputSize)

   local gradOutput1 = gradOutputs[2]:clone()
   for step=1,nStep do
      outputs[step] = lstm:forward(inputs[step])
   end

   local gradInputs72 = torch.Tensor(nStep, batchSize, inputSize)
   for step=nStep,1,-1 do
      gradInputs72[step] = lstm:backward(inputs[step], gradOutputs[step])
   end

   local lstm3 = nn.Sequencer(lstm2)
   lstm3:zeroGradParameters()
   local outputs3 = lstm3:forward(inputs)
   local gradInputs3 = lstm3:backward(inputs, gradOutputs)
   mytester:assert(outputs3:size(1) == outputs:size(1), "Sequencer LSTM output size err")
   mytester:assert(gradInputs3:size(1) == gradInputs72:size(1), "Sequencer LSTM gradInputs size err")
   for step=1,nStep do
      mytester:assertTensorEq(outputs3[step], outputs[step], 0.00001, "Sequencer LSTM output "..step)
      mytester:assertTensorEq(gradInputs3[step], gradInputs72[step], 0.00001, "Sequencer LSTM gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[2], gradOutput1, 0.00001, "Sequencer lstm gradOutput modified error")

   -- test remember modes : 'both', 'eval' for training(), evaluate(), training()
   local lstm = nn.LSTM(5,5)
   local seq = nn.Sequencer(lstm)
   local inputTrain = torch.randn(3,5)
   local inputEval = torch.randn(1,5)

   -- this shouldn't fail
   local modes = {'both', 'eval'}
   for i, mode in ipairs(modes) do
     seq:remember(mode)

     -- do one epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)

     -- evaluate
     seq:evaluate()
     seq:forward(inputEval)

     -- do another epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)
   end
end

function rnntest.Sequencer_tensoreval()
   -- test that it behave the same in evaluation
   local seqlen, batchsize, fsize = 5, 3, 4
   local input = torch.randn(seqlen, batchsize, fsize)
   local lstm = nn.FastLSTM(fsize, fsize)
   local lstm2 = lstm:clone()
   local seq = nn.Sequencer(lstm)
   local seq2 = nn.Sequential()
            :add(nn.SplitTable(1))
            :add(nn.Sequencer(lstm2))
   seq:evaluate()
   seq2:evaluate()
   local output = seq:forward(input)
   local output2 = seq2:forward(input)
   for i=1,seqlen do
      mytester:assertTensorEq(output[i], output2[i], 0.000001)
   end
   seq:forget()
   seq2:forget()
   -- test eval after forget
   local input = torch.randn(seqlen, batchsize, fsize)
   local output = seq:forward(input)
   local output2 = seq2:forward(input)
   for i=1,seqlen do
      mytester:assertTensorEq(output[i], output2[i], 0.000001)
   end
   -- test eval after forget + variable size
   for i=1,3 do
      seqlen, batchsize = math.random(2,7), math.random(2,7)
      local input = torch.randn(seqlen, batchsize, fsize)
      local output = seq:forward(input)
      local output2 = seq2:forward(input)
      for i=1,seqlen do
         mytester:assertTensorEq(output[i], output2[i], 0.000001)
      end
   end
   -- test again with remember
   seq:remember()
   seq2:remember()
   local input = torch.randn(seqlen, batchsize, fsize)
   local outputs = seq:forward(input)
   local outputs2 = seq2:forward(input)
   for i=1,seqlen do
      mytester:assertTensorEq(output[i], output2[i], 0.000001)
   end
   for i=1,3 do
      local seqlen = math.random(2,7)
      local input = torch.randn(seqlen, batchsize, fsize)
      local outputs = seq:forward(input)
      local outputs2 = seq2:forward(input)
      for i=1,seqlen do
         mytester:assertTensorEq(output[i], output2[i], 0.000001)
      end
   end
end

function rnntest.BiSequencer()
   local hiddenSize = 8
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
   local brnn = nn.BiSequencer(fwd:clone(), bwd:clone())
   local inputs, gradOutputs = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize*2)
   end
   local outputs = brnn:forward(inputs)
   local gradInputs = brnn:backward(inputs, gradOutputs)
   mytester:assert(#inputs == #outputs, "BiSequencer #outputs error")
   mytester:assert(#inputs == #gradInputs, "BiSequencer #outputs error")

   -- forward
   local fwdSeq = nn.Sequencer(fwd)
   local bwdSeq = nn.Sequencer(bwd)
   local zip, join = nn.ZipTable(), nn.Sequencer(nn.JoinTable(1,1))
   local fwdOutputs = fwdSeq:forward(inputs)
   local bwdOutputs = _.reverse(bwdSeq:forward(_.reverse(inputs)))
   local zipOutputs = zip:forward{fwdOutputs, bwdOutputs}
   local outputs2 = join:forward(zipOutputs)
   for i,output in ipairs(outputs) do
      mytester:assertTensorEq(output, outputs2[i], 0.000001, "BiSequencer output err "..i)
   end

   -- backward
   local joinGradInputs = join:backward(zipOutputs, gradOutputs)
   local zipGradInputs = zip:backward({fwdOutputs, bwdOutputs}, joinGradInputs)
   local bwdGradInputs = _.reverse(bwdSeq:backward(_.reverse(inputs), _.reverse(zipGradInputs[2])))
   local fwdGradInputs = fwdSeq:backward(inputs, zipGradInputs[1])
   local gradInputs2 = zip:forward{fwdGradInputs, bwdGradInputs}
   for i,gradInput in ipairs(gradInputs) do
      local gradInput2 = gradInputs2[i]
      gradInput2[1]:add(gradInput2[2])
      mytester:assertTensorEq(gradInput, gradInput2[1], 0.000001, "BiSequencer gradInput err "..i)
   end

   -- params
   local brnn2 = nn.Sequential():add(fwd):add(bwd)
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   mytester:assert(#params == #params2, "BiSequencer #params err")
   mytester:assert(#params == #gradParams, "BiSequencer #gradParams err")
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams err "..i)
   end

   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters()
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params update err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams zero err "..i)
   end
end

function rnntest.BiSequencerLM()
   local hiddenSize = 8
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
   local brnn = nn.BiSequencerLM(fwd:clone(), bwd:clone())
   local inputs, gradOutputs = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize*2)
   end
   local outputs = brnn:forward(inputs)
   local gradInputs = brnn:backward(inputs, gradOutputs)
   mytester:assert(#inputs == #outputs, "BiSequencerLM #outputs error")
   mytester:assert(#inputs == #gradInputs, "BiSequencerLM #outputs error")

   -- forward
   local fwdSeq = nn.Sequencer(fwd)
   local bwdSeq = nn.Sequencer(bwd)
   local merge = nn.Sequential():add(nn.ZipTable()):add(nn.Sequencer(nn.JoinTable(1,1)))

   local fwdOutputs = fwdSeq:forward(_.first(inputs, #inputs-1))
   local bwdOutputs = _.reverse(bwdSeq:forward(_.reverse(_.last(inputs, #inputs-1))))

   local fwdMergeInputs = _.clone(fwdOutputs)
   table.insert(fwdMergeInputs, 1, fwdOutputs[1]:clone():zero())
   local bwdMergeInputs = _.clone(bwdOutputs)
   table.insert(bwdMergeInputs, bwdOutputs[1]:clone():zero())

   local outputs2 = merge:forward{fwdMergeInputs, bwdMergeInputs}

   for i,output in ipairs(outputs) do
      mytester:assertTensorEq(output, outputs2[i], 0.000001, "BiSequencerLM output err "..i)
   end

   -- backward
   local mergeGradInputs = merge:backward({fwdMergeInputs, bwdMergeInputs}, gradOutputs)

   local bwdGradInputs = _.reverse(bwdSeq:backward(_.reverse(_.last(inputs, #inputs-1)), _.reverse(_.first(mergeGradInputs[2], #inputs-1))))
   local fwdGradInputs = fwdSeq:backward(_.first(inputs, #inputs-1), _.last(mergeGradInputs[1], #inputs-1))

   for i,gradInput in ipairs(gradInputs) do
      local gradInput2
      if i == 1 then
         gradInput2 = fwdGradInputs[1]
      elseif i == #inputs then
         gradInput2 = bwdGradInputs[#inputs-1]
      else
         gradInput2 = fwdGradInputs[i]:clone()
         gradInput2:add(bwdGradInputs[i-1])
      end
      mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "BiSequencerLM gradInput err "..i)
   end

   -- params
   local brnn2 = nn.Sequential():add(fwd):add(bwd)
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   mytester:assert(#params == #params2, "BiSequencerLM #params err")
   mytester:assert(#params == #gradParams, "BiSequencerLM #gradParams err")
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencerLM params err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencerLM gradParams err "..i)
   end

   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters()
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencerLM params update err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencerLM gradParams zero err "..i)
   end
end

function rnntest.Repeater()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5
   local rnn = nn.LinearRNN(inputSize, outputSize)
   local rnn2 = rnn:clone()

   local inputs, outputs, gradOutputs = {}, {}, {}
   local input = torch.randn(batchSize, inputSize)
   for step=1,nStep do
      outputs[step] = rnn:forward(input)
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end
   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = rnn:backward(input, gradOutputs[step])
   end

   local rnn3 = nn.Repeater(rnn2, nStep)
   local outputs3 = rnn3:forward(input)
   local gradInput3 = rnn3:backward(input, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Repeater output size err")
   mytester:assert(#outputs3 == #gradInputs, "Repeater gradInputs size err")
   local gradInput = gradInputs[1]:clone():zero()
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Repeater output "..step)
      gradInput:add(gradInputs[step])
   end
   mytester:assertTensorEq(gradInput3, gradInput, 0.00001, "Repeater gradInput err")

   -- test with Recursor
   local rnn = nn.LinearRNN(inputSize, outputSize)
   local rnn2 = rnn:clone()

   local rnn3 = nn.Repeater(rnn, nStep)
   local rnn4 = nn.Repeater(nn.Sequential():add(nn.Identity()):add(rnn2), nStep)

   rnn3:zeroGradParameters()
   rnn4:zeroGradParameters()

   local outputs = rnn3:forward(input)
   local outputs2 = rnn4:forward(input)

   local gradInput = rnn3:backward(input, gradOutputs)
   local gradInput2 = rnn4:backward(input, gradOutputs)

   mytester:assert(#outputs == #outputs2, "Repeater output size err")
   for i=1,#outputs do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Repeater(Recursor) output err")
   end
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "Repeater(Recursor) gradInput err")

   rnn3:updateParameters(1)
   rnn4:updateParameters(1)

   local params, gradParams = rnn3:parameters()
   local params2, gradParams2 = rnn4:parameters()

   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Repeater(Recursor) param err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Repeater(Recursor) gradParam err "..i)
   end
end

function rnntest.SequencerCriterion()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5
   -- https://github.com/Element-Research/rnn/issues/128
   local criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion())
   criterion.v2 = false
   local sc = nn.SequencerCriterion(criterion:clone())
   local input = {}
   local target = {}
   local err2 = 0
   local gradInput2 = {}
   for i=1,nStep do
      input[i] = torch.randn(batchSize, inputSize)
      target[i] = torch.randperm(inputSize):narrow(1,1,batchSize)
      err2 = err2 + criterion:forward(input[i], target[i])
      gradInput2[i] = criterion:backward(input[i], target[i]):clone()
   end
   local err = sc:forward(input, target)
   mytester:assert(math.abs(err-err2) < 0.000001, "SequencerCriterion forward err")
   local gradInput = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i], gradInput2[i], 0.000001, "SequencerCriterion backward err "..i)
   end

   -- test type()
   sc.gradInput = {}
   sc:float()

   for i=1,nStep do
      input[i] = input[i]:float()
      target[i] = target[i]:float()
   end

   local err3 = sc:forward(input, target)
   mytester:assert(math.abs(err - err3) < 0.000001, "SequencerCriterion forward type err")
   local gradInput3 = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i]:float(), gradInput3[i], 0.000001, "SequencerCriterion backward type err "..i)
   end

   -- Test tensor as input
   local sc2 = sc:clone()
   local split = nn.SplitTable(1)
   local input = torch.randn(nStep, batchSize, inputSize):float()
   local target = torch.Tensor(nStep, batchSize):float()
   for i=1,nStep do
     target[i] = torch.randperm(inputSize):narrow(1,1,batchSize)
   end
   local errTensorInput = sc:forward(input, target) -- As Tensor
   local errTableInput = sc2:forward(split:forward(input), split:forward(target)) -- As Table
   mytester:assert(math.abs(errTensorInput - errTableInput) == 0, "SequencerCriterion forward type err")
   local gradInputTensor = sc:backward(input, target)
   local gradInputTable = sc:backward(split:forward(input), split:forward(target))
   mytester:assertTensorEq(gradInputTensor, torch.cat(gradInputTable, 1):view(gradInputTensor:size()), 0, "SequencerCriterion backward type err ")

   if pcall(function() require 'cunn' end) then
      -- test cuda()
      sc.gradInput = {}
      sc:cuda()

      local gradInput4 = {}
      input = input:cuda()
      target = target:cuda()

      local err4 = sc:forward(input, target)
      mytester:assert(math.abs(errTensorInput - err4) < 0.000001, "SequencerCriterion forward cuda err")
      local gradInput4 = sc:backward(input, target)
      for i=1,nStep do
         mytester:assertTensorEq(gradInput4[i]:float(), gradInput3[i], 0.000001, "SequencerCriterion backward cuda err "..i)
      end
   end
end

function rnntest.RepeaterCriterion()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5
   local criterion = nn.ClassNLLCriterion()
   local sc = nn.RepeaterCriterion(criterion:clone())
   local input = {}
   local target = torch.randperm(inputSize):narrow(1,1,batchSize)
   local err2 = 0
   local gradInput2 = {}
   for i=1,nStep do
      input[i] = torch.randn(batchSize, inputSize)
      err2 = err2 + criterion:forward(input[i], target)
      gradInput2[i] = criterion:backward(input[i], target):clone()
   end
   local err = sc:forward(input, target)
   mytester:assert(math.abs(err-err2) < 0.000001, "RepeaterCriterion forward err")
   local gradInput = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i], gradInput2[i], 0.000001, "RepeaterCriterion backward err "..i)
   end

   -- test type()
   sc:float()

   local gradInput3 = {}
   target = target:float()
   for i=1,nStep do
      input[i] = input[i]:float()
   end

   local err3 = sc:forward(input, target)
   mytester:assert(math.abs(err - err3) < 0.000001, "RepeaterCriterion forward type err")
   local gradInput3 = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i]:float(), gradInput3[i], 0.000001, "RepeaterCriterion backward type err "..i)
   end

   -- Test tensor as input
   sc:double()
   local sc2 = sc:clone()
   local split = nn.SplitTable(1)
   local input = torch.randn(nStep, batchSize, inputSize)
   local target = torch.randperm(inputSize):narrow(1,1,batchSize)
   local errTensorInput = sc:forward(input, target) -- As Tensor
   local errTableInput = sc2:forward(split:forward(input), target) -- As Table
   mytester:assert(math.abs(errTensorInput - errTableInput) == 0, "RepeaterCriterion forward type err")
   local gradInputTensor = sc:backward(input, target)
   local gradInputTable = sc:backward(split:forward(input), target)
   mytester:assertTensorEq(gradInputTensor, torch.cat(gradInputTable, 1):view(gradInputTensor:size()), 0, "RepeaterCriterion backward type err ")

end

function rnntest.RecLSTM_checkgrad()
   if not pcall(function() require 'optim' end) then return end

   local hiddenSize = 2
   local nIndex = 2
   local r = nn.RecLSTM(hiddenSize, hiddenSize)

   local rnn = nn.Sequential()
   rnn:add(r)
   rnn:add(nn.Linear(hiddenSize, nIndex))
   rnn:add(nn.LogSoftMax())
   rnn = nn.Recursor(rnn)

   local criterion = nn.ClassNLLCriterion()
   local inputs = torch.randn(3, 4, 2)
   local targets = torch.Tensor(3,4):random(1,2)
   local parameters, grads = rnn:getParameters()

   function f(x)
      parameters:copy(x)

      -- Do the forward prop
      rnn:zeroGradParameters()
      local err = 0
      local outputs = {}
      for i = 1, inputs:size(1) do
         outputs[i] = rnn:forward(inputs[i])
         err = err + criterion:forward(outputs[i], targets[i])
      end
      for i = inputs:size(1), 1, -1 do
         local gradOutput = criterion:backward(outputs[i], targets[i])
         rnn:backward(inputs[i], gradOutput)
      end
      rnn:forget()
      return err, grads
   end

   local err = optim.checkgrad(f, parameters:clone())
   mytester:assert(err < 0.0001, "RecLSTM optim.checkgrad error")
end

function rnntest.Recursor()
   local batchSize = 4
   local inputSize = 3
   local hiddenSize = 12
   local outputSize = 7
   local seqlen = 5

   local inputs = torch.randn(seqlen, batchSize, inputSize)
   local gradOutputs = torch.randn(seqlen, batchSize, outputSize)

   -- USE CASE 1. Recursor(LSTM)

   local rnn = nn.RecLSTM(inputSize, outputSize)
   local re2 = rnn:clone()
   local re = nn.Recursor(nn.Sequential():add(rnn))
   re:zeroGradParameters()
   re2:zeroGradParameters()

   local outputs = torch.Tensor(seqlen, batchSize, outputSize)
   local outputs2 = outputs:clone()
   local gradInputs = torch.Tensor(seqlen, batchSize, inputSize)

   for i=1,seqlen do
      -- forward
      outputs[i] = re:forward(inputs[i])
      outputs2[i] = re2:forward(inputs[i])
   end

   local gradInputs_2 = torch.Tensor(seqlen, batchSize, inputSize)
   for i=seqlen,1,-1 do
      -- backward
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
      gradInputs[i] = re:backward(inputs[i], gradOutputs[i])
   end

   mytester:assertTensorEq(outputs, outputs2, 0.0000001, "Recursor(LSTM) fwd err ")
   mytester:assertTensorEq(gradInputs, gradInputs_2, 0.0000001, "Recursor(LSTM) bwd err ")

   re2:updateParameters(0.1)
   re:updateParameters(0.1)

   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = re2:parameters()

   mytester:assert(#params == #params2, "Recursor(LSTM) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recursor(LSTM) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recursor(LSTM) accGradParams err "..i)
   end

   -- USE CASE 2. Sequencer(Recursor)

   local re2 = nn.RecLSTM(inputSize, outputSize)
   local lstm2 = re2:clone()
   local rec = nn.Recursor(lstm2)
   local seq = nn.Sequencer(rec)
   mytester:assert(not rec.copyInputs)
   mytester:assert(not rec.copyGradOutputs)
   mytester:assert(not lstm2.copyInputs)
   mytester:assert(not lstm2.copyGradOutputs)

   seq:zeroGradParameters()
   re2:zeroGradParameters()

   local outputs = seq:forward(inputs)
   local gradInputs = seq:backward(inputs, gradOutputs)

   for i=1,seqlen do
      outputs2[i] = re2:forward(inputs[i])
   end

   for i=seqlen,1,-1 do
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end

   re2:updateParameters(0.1)

   mytester:assertTensorEq(outputs, outputs2, 0.0000001, "Sequencer(Recursor(LSTM)) fwd err ")
   mytester:assertTensorEq(gradInputs, gradInputs_2, 0.0000001, "Sequencer(Recursor(LSTM)) bwd err ")

   seq:updateParameters(0.1)

   local params, gradParams = seq:parameters()
   local params2, gradParams2 = re2:parameters()

   mytester:assert(#params == #params2, "Sequencer(Recursor(LSTM)) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Sequencer(Recursor(LSTM)) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Sequencer(Recursor(LSTM)) accGradParams err "..i)
   end

   -- USE CASE 3. Recursor(Recursor(LSTM))

   local rnn = nn.RecLSTM(inputSize, outputSize)
   local re2 = rnn:clone()
   local re = nn.Recursor(nn.Recursor(nn.Sequential():add(rnn)))
   re:zeroGradParameters()
   re2:zeroGradParameters()

   for i=1,seqlen do
      -- forward
      outputs = re:forward(inputs[i])
      outputs2 = re2:forward(inputs[i])
   end

   for i=seqlen,1,-1 do
      -- backward
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end

   re2:updateParameters(0.1)

   -- recursor requires reverse-time-step order during backward
   for i=seqlen,1,-1 do
      gradInputs[i] = re:backward(inputs[i], gradOutputs[i])
   end

   mytester:assertTensorEq(outputs, outputs2, 0.0000001, "Recursor(Recursor(LSTM)) fwd err ")
   mytester:assertTensorEq(gradInputs, gradInputs_2, 0.0000001, "Recursor(Recursor(LSTM)) bwd err ")

   re:updateParameters(0.1)

   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = re2:parameters()

   mytester:assert(#params == #params2, "Recursor(Recursor(LSTM)) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recursor(Recursor(LSTM)) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recursor(Recursor(LSTM)) accGradParams err "..i)
   end

end

function rnntest.Recurrence()
   local batchSize = 4
   local inputSize = 3
   local outputSize = 2
   local seqlen = 3

   local lstm2 = nn.RecLSTM(inputSize, outputSize)
   local seq2 = nn.Sequencer(lstm2)

   local stepmodule = nn.Sequential()
      :add(nn.FlattenTable())
      :add(lstm2.modules[1]:clone())
   local recmodule = nn.Sequential()
      :add(nn.Recurrence(stepmodule, {{outputSize}, {outputSize}}, 1, seqlen))
      :add(nn.SelectTable(1))
   local lstm = recmodule
   local seq = nn.Sequencer(lstm)

   local inputs = torch.randn(seqlen, batchSize, inputSize)
   local gradOutputs = torch.randn(seqlen, batchSize, outputSize)

   seq:zeroGradParameters()
   seq2:zeroGradParameters()

   local params, gradParams = seq:parameters()
   local params2, gradParams2 = seq2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recurrence updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recurrence accGradParams err "..i)
   end

   local outputs = seq:forward(inputs)
   local outputs2 = seq2:forward(inputs)

   mytester:assertTensorEq(outputs, outputs2, 0.0000001, "Recurrence fwd err ")

   local gradInputs = seq:backward(inputs, gradOutputs)
   local gradInputs2 = seq2:backward(inputs, gradOutputs)

   mytester:assertTensorEq(gradInputs, gradInputs2, 0.0000001, "Recurrence bwd err ")

   seq:updateParameters(0.1)
   seq2:updateParameters(0.1)

   mytester:assert(#params == #params2, "Recurrence #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recurrence updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recurrence accGradParams err "..i)
   end
end

function rnntest.Recurrence_nested()
   -- issue 107
   -- this will test the use case where an AbstractRecurrent.modules[1]
   -- contains an AbstractRecurrent instance!

   local batchSize = 4
   local hiddenSize = 2
   local seqlen = 3

   local lstm = nn.RecLSTM(hiddenSize,hiddenSize)

   local rm = nn.Sequential()
      :add(nn.CSubTable())
      :add(lstm)
      :add(nn.Linear(hiddenSize,hiddenSize))
      :add(nn.Sigmoid())

   local rnn = nn.Recurrence(rm, hiddenSize, 1)

   local seq = nn.Sequencer(rnn)

   local inputs, gradOutputs = {}, {}
   for i=1,seqlen do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize)
   end

   for n=1,3 do
      seq:evaluate()
      seq:training()
      seq:zeroGradParameters()

      seq:forward(inputs)
      seq:backward(inputs, gradOutputs)

      mytester:assert(rnn.step == 4)
      mytester:assert(lstm.step == 4)
   end
end

-- mock Recurrent and LSTM stepmodule for UT
-- must be stateless
-- forwarding zeros must not return zeros -> use Sigmoid()
local function recurrentModule()
   local recurrent = nn.Sequential()
   local parallel = nn.ParallelTable()
   parallel:add(nn.Sigmoid()); parallel:add(nn.Identity())
   recurrent = nn.Sequential()
   recurrent:add(parallel)
   recurrent:add(nn.SelectTable(1))
   return recurrent
end

local function lstmModule()
   local recurrent = nn.Sequential()
   local parallel = nn.ParallelTable()
   parallel:add(nn.Sigmoid()); parallel:add(nn.Identity()); parallel:add(nn.Identity())
   recurrent = nn.Sequential()
   recurrent:add(parallel)
   recurrent:add(nn.NarrowTable(1, 2))
   return recurrent
end

local function firstElement(a)
   return torch.type(a) == 'table' and a[1] or a
end

function rnntest.MaskZero_main()
   local recurrents = {['recurrent'] = recurrentModule(), ['lstm'] = lstmModule()}
   -- Note we use lstmModule input signature and firstElement to prevent duplicate code
   for name, recurrent in pairs(recurrents) do
      -- test encapsulated module first
      -- batch
      local i = torch.rand(5, 10)
      local e = nn.Sigmoid():forward(i)
      local o = firstElement(recurrent:forward({i, torch.zeros(5, 10), torch.zeros(5, 10)}))
      mytester:assertlt(torch.norm(o - e), precision, 'mock ' .. name .. ' module failed for batch')

      -- test mask zero module now
      local module = nn.MaskZero(recurrent)
      module:setZeroMask(false)
      -- batch forward
      local i = torch.rand(5, 10)
      local e = firstElement(recurrent:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertgt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(e - o), precision, 'error on batch forward for ' .. name)
      local i = torch.zeros(5, 10)
      module:setZeroMask(torch.ByteTensor(5):fill(1))
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertlt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      local i = torch.Tensor({{0, 0, 0}, {1, 2, 5}})
      module:setZeroMask(torch.ByteTensor({1,0}))
      -- clone r because it will be update by module:forward call
      local r = firstElement(recurrent:forward({i, torch.rand(2, 3), torch.rand(2, 3)})):clone()
      local o = firstElement(module:forward({i, torch.rand(2, 3), torch.rand(2, 3)}))
      mytester:assertgt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)
      r[1]:zero()
      mytester:assertlt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)

      -- check gradients
      local jac = nn.Jacobian
      local sjac = nn.SparseJacobian
      -- Note: testJacobian doesn't support table inputs or outputs
      -- Use a SplitTable and SelectTable to adapt module
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.MaskZero(recurrent))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end

      local input = torch.rand(name == 'lstm' and 3 or 2, 10)
      module:setZeroMask(false)
      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error on state for ' .. name)
      -- IO
      function module.clearState(self) return self end
      local ferr,berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for ' .. name)
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for ' .. name)
      -- batch
      -- rebuild module to avoid correlated tests
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.MaskZero(recurrent))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end
      module:setZeroMask(false)

      local input = torch.rand(name == 'lstm' and 3 or 2, 5, 10)
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err, precision, 'batch error on state for ' .. name)

      -- full test on convolution and linear modules
      local module = nn.Sequential() :add( nn.ParallelTable() :add(nn.SpatialConvolution(1,2,3,3)) :add(nn.Linear(100,2)) )
      --module = module:float()
      local batchNum = 5
      local input = {torch.rand(batchNum,1,10,10), torch.rand(batchNum,100)}
      local zeroRowNum = 2
      for i = 1,#input do
         input[i]:narrow(1,1,zeroRowNum):zero()
      end
      --module = nn.MaskZero(module, 3)
      local output = module:forward(input)
      for i = 1,#input do
         for j = 1,batchNum do
            local rmi = input[i][j]:view(-1) -- collapse dims
            local vectorDim = rmi:dim()
            local rn = rmi.new()
            rn:norm(rmi, 2, vectorDim)
            local err = rn[1]
            if j<=zeroRowNum then
               -- check zero outputs
               mytester:assertlt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            else
               -- check non-zero outputs
               mytester:assertgt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            end
         end
      end
   end
end

function rnntest.nn_utils_getZeroMask()
   local sequence = torch.randn(3,4,2)
   sequence[{1,2}]:fill(0)
   local zeroMask = nn.utils.getZeroMaskSequence(sequence)
   mytester:assert(torch.type(zeroMask) == 'torch.ByteTensor')
   local zeroMask2 = torch.Tensor(3,4):fill(0)
   zeroMask2[{1,2}] = 1
   mytester:assertTensorEq(zeroMask:type(torch.type(zeroMask2)), zeroMask2, 0.00000001)
end

function rnntest.AbstractRecurrent_maskZero()
   local inputs = {}

   local input = torch.zeros(4,4,10)
   local sequence = torch.randn(4,10)
   input:select(2,1):select(1,4):copy(sequence[1])
   input:select(2,2):narrow(1,3,2):copy(sequence:narrow(1,1,2))
   input:select(2,3):narrow(1,2,3):copy(sequence:narrow(1,1,3))
   input:select(2,4):copy(sequence)


   local function testmask(rnn)
      local seq = nn.Sequencer(rnn:maskZero())

      local zeroMask = nn.utils.getZeroMaskSequence(input)
      seq:setZeroMask(zeroMask)
      local outputs = seq:forward(input)

      mytester:assert(math.abs(outputs[1]:narrow(1,1,3):sum()) < 0.0000001, torch.type(rnn).." mask zero 1 err")
      mytester:assert(math.abs(outputs[2]:narrow(1,1,2):sum()) < 0.0000001, torch.type(rnn).." mask zero 2 err")
      mytester:assert(math.abs(outputs[3]:narrow(1,1,1):sum()) < 0.0000001, torch.type(rnn).." mask zero 3 err")

      mytester:assertTensorEq(outputs[1][4], outputs[2][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[3][2], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[4][1], 0.0000001, torch.type(rnn).." mask zero err")

      mytester:assertTensorEq(outputs[2][4], outputs[3][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[2][4], outputs[4][2], 0.0000001, torch.type(rnn).." mask zero err")

      mytester:assertTensorEq(outputs[3][4], outputs[4][3], 0.0000001, torch.type(rnn).." mask zero err")
   end

   testmask(nn.LinearRNN(10, 10))
   testmask(nn.RecLSTM(10, 10))
   testmask(nn.RecGRU(10, 10))
end

local function forwardbackward(module, criterion, input, expected)
  local output = module:forward(input)
  criterion:forward(output, expected)
  module:zeroGradParameters()
  module:backward(input, criterion:backward(output, expected))
  module:updateParameters(1)
  return output
end

function rnntest.LookupTableMaskZero()
   local batchSize = math.random(5, 10)
   local outputSize = math.random(5, 10)
   local indexSize = batchSize

   local m1 = nn.LookupTable(indexSize, outputSize)
   local m2 = nn.LookupTableMaskZero(indexSize, outputSize)
   m2.weight:narrow(1, 2, indexSize):copy(m1.weight)
   local criterion = nn.MSECriterion()
   -- Zero padding will change averaging
   criterion.sizeAverage = false

   -- verify that LookupTables have the same results (modulo zero padding)
   -- through multiple backpropagations
   for i=1, 10 do
      local input1 = torch.randperm(batchSize)
      local input2 = torch.zeros(batchSize + 2)
      input2:narrow(1, 1, batchSize):copy(input1)
      local expected1 = torch.rand(batchSize, outputSize)
      local expected2 = torch.rand(batchSize + 2, outputSize)
      expected2:narrow(1, 1, batchSize):copy(expected1)
      local o1 = forwardbackward(m1, criterion, input1, expected1)
      local o2 = forwardbackward(m2, criterion, input2, expected2)
      -- output modulo zero index should be the same
      mytester:assertlt(torch.norm(o1 - o2:narrow(1, 1, batchSize), 2), precision)
      -- zero index should yield zero vector
      mytester:assertlt(o2[batchSize + 1]:norm(2), precision)
      mytester:assertlt(o2[batchSize + 2]:norm(2), precision)
      -- weights should be equivalent
      mytester:assertlt(torch.norm(m1.weight - m2.weight:narrow(1, 2, indexSize), 2), precision)
  end
end

function rnntest.MaskZeroCriterion()
   local function testMaskZeroCriterion(v2)
      local batchSize = 3
      local nClass = 10
      local input = torch.randn(batchSize, nClass)
      local target = torch.LongTensor(batchSize):random(1,nClass)

      local nll = nn.ClassNLLCriterion()
      local mznll = nn.MaskZeroCriterion(nll, 1)
      mznll.v2 = v2

      -- test that it works when nothing to mask
      if v2 then
         mznll:setZeroMask(false)
      end
      local err = mznll:forward(input, target)
      local gradInput = mznll:backward(input, target):clone()

      local err2 = nll:forward(input, target)
      local gradInput2 = nll:backward(input, target)

      mytester:assert(math.abs(err - err2) < 0.0000001, "MaskZeroCriterion No-mask fwd err")
      mytester:assertTensorEq(gradInput, gradInput2, 0.0000001, "MaskZeroCriterion No-mask bwd err")

      -- test that it works when last row to mask
      local zeroMask
      if v2 then
         zeroMask = torch.ByteTensor(batchSize):zero()
         zeroMask[batchSize] = 1
         mznll:setZeroMask(zeroMask)
      else
         input[batchSize]:zero()
         target[batchSize] = 0
      end

      local err = mznll:forward(input, target)
      local gradInput = mznll:backward(input, target):clone()

      local input2 = input:narrow(1,1,batchSize-1)
      local target2 = target:narrow(1,1,batchSize-1)
      local err2 = nll:forward(input2, target2)
      local gradInput2 = nll:backward(input2, target2)

      mytester:assert(gradInput[batchSize]:sum() == 0, "MaskZeroCriterion last-mask bwd zero err")
      mytester:assert(math.abs(err - err2) < 0.0000001, "MaskZeroCriterion last-mask fwd err")
      mytester:assertTensorEq(gradInput:narrow(1,1,batchSize-1), gradInput2, 0.0000001, "MaskZeroCriterion last-mask bwd err")

      -- test type-casting
      mznll:float()
      local input3 = input:float()
      if v2 then
         mznll:setZeroMask(zeroMask)
      end
      local err3 = mznll:forward(input3, target)
      local gradInput3 = mznll:backward(input3, target):clone()

      mytester:assert(math.abs(err3 - err) < 0.0000001, "MaskZeroCriterion cast fwd err")
      mytester:assertTensorEq(gradInput3, gradInput:float(), 0.0000001, "MaskZeroCriterion cast bwd err")

      if pcall(function() require 'cunn' end) then
         -- test cuda
         mznll:cuda()
         if v2 then
            mznll:setZeroMask(zeroMask:cudaByte())
         end
         local input4 = input:cuda()
         local target4 = target:cuda()
         local err4 = mznll:forward(input4, target4)
         local gradInput4 = mznll:backward(input4, target4):clone()

         mytester:assert(math.abs(err4 - err) < 0.0000001, "MaskZeroCriterion cuda fwd err "..tostring(v2))
         mytester:assertTensorEq(gradInput4:float(), gradInput3, 0.0000001, "MaskZeroCriterion cuda bwd err "..tostring(v2))
      end

      -- issue 128
      local input, target = torch.zeros(3,2), torch.Tensor({1,2,1}) -- batch size 3, 2 classes
      local crit = nn.MaskZeroCriterion(nn.ClassNLLCriterion())
      crit.v2 = v2
      if v2 then
         zeroMask:fill(1)
         crit:setZeroMask(zeroMask)
      end

      -- output from a masked module gives me all zeros
      local loss = crit:forward(input, target)
      mytester:assert(crit.isEmptyBatch)
      mytester:assert(loss == 0, "MaskZeroCriterion all zeros fwd err "..tostring(v2))

      local gradInput = crit:backward(input, target)
      mytester:assert(gradInput:sum() == 0, "MaskZeroCriterion all zeros bwd err "..tostring(v2))

      -- test table input
      local inputSize = 5
      local input = {torch.randn(batchSize, inputSize), torch.randn(batchSize, inputSize)}
      local target = torch.randn(batchSize):fill(1)

      local criterion = nn.MaskZeroCriterion(nn.CosineEmbeddingCriterion(), 1)
      criterion.v2 = v2
      if v2 then
         zeroMask:zero()
         zeroMask[2] = 1
         criterion:setZeroMask(zeroMask)
      else
         input[1][2]:zero()
      end

      local loss = criterion:forward(input, target)
      local gradInput = criterion:backward(input, target)
      mytester:assert(gradInput[1][2]:sum() + gradInput[2][2]:sum() == 0)
   end

   testMaskZeroCriterion(false)
   testMaskZeroCriterion(true)
end

function rnntest.MaskZero_where()
   local hiddensize = 5
   local batchsize = 4
   local seqlen = 7

   local rnn = nn.LinearRNN(hiddensize, hiddensize):maskZero()
   rnn = nn.Sequencer(rnn)

   -- is there any difference between start and end padding?

   local inputs, gradOutputs = {}, {}

   for i=1,seqlen do
      inputs[i] = torch.randn(batchsize, hiddensize)
      gradOutputs[i] = torch.randn(batchsize, hiddensize)
   end

   local zeroMask = torch.ByteTensor(seqlen, batchsize):zero()
   zeroMask[1]:fill(1)
   rnn:setZeroMask(zeroMask)
   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)

   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = {}, {}
   for i=1,#params do
      params2[i] = params[i]:clone()
      gradParams2[i] = gradParams[i]:clone()
   end

   local outputs2, gradInputs2 = {}, {}
   for i=1,seqlen do
      outputs2[i] = outputs[i]:clone()
      gradInputs2[i] = gradInputs[i]:clone()
   end
   inputs[seqlen] = table.remove(inputs, 1)
   gradOutputs[seqlen] = table.remove(gradOutputs, 1)

   zeroMask:zero()[seqlen]:fill(1)
   rnn:setZeroMask(zeroMask)
   rnn:forget()
   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)

   for i=1,seqlen-1 do
      mytester:assertTensorEq(outputs[i], outputs2[i+1], 0.000001)
      mytester:assertTensorEq(gradInputs[i], gradInputs2[i+1], 0.000001)
   end

   for i=1,#params do
      mytester:assertTensorEq(gradParams2[i], gradParams[i], 0.000001)
   end

   -- how about in the middle? is it the same as a forget() in between

   local inputs, gradOutputs = {}, {}

   for i=1,seqlen do
      inputs[i] = torch.randn(batchsize, hiddensize)
      gradOutputs[i] = torch.randn(batchsize, hiddensize)
   end

   local zeroMask = torch.ByteTensor(seqlen, batchsize):zero()
   zeroMask[4]:fill(1)
   rnn:setZeroMask(zeroMask)
   rnn:forget()
   local rnn2 = rnn:clone()

   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)

   local _ = require 'moses'
   local inputs1 = _.first(inputs, 3)
   local gradOutputs1 = _.first(gradOutputs, 3)

   local zeroMask = torch.ByteTensor(3, batchsize):zero()
   rnn2:setZeroMask(zeroMask)
   local outputs1 = rnn2:forward(inputs1)
   rnn2:zeroGradParameters()
   local gradInputs1 = rnn2:backward(inputs1, gradOutputs1)

   for i=1,3 do
      mytester:assertTensorEq(outputs[i], outputs1[i], 0.000001)
      mytester:assertTensorEq(gradInputs[i], gradInputs1[i], 0.000001)
   end

   rnn2:forget() -- forget at mask zero

   local inputs2 = _.last(inputs, 3)
   local gradOutputs2 = _.last(gradOutputs, 3)

   rnn2:setZeroMask(zeroMask)
   local outputs2 = rnn2:forward(inputs2)
   local gradInputs2 = rnn2:backward(inputs2, gradOutputs2)

   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = rnn2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(gradParams2[i], gradParams[i], 0.000001)
   end

   for i=1,3 do
      mytester:assertTensorEq(outputs[i+4], outputs2[i], 0.000001)
      mytester:assertTensorEq(gradInputs[i+4], gradInputs2[i], 0.000001)
   end
end

function rnntest.issue129()
   -- test without rnn
   local model1 = nn.Sequential()
   model1:add(nn.SpatialBatchNormalization(2))

   local input = torch.randn(4, 2, 64, 64)  -- batch_size X channels X height X width

   model1:training()
   local output
   for i=1, 1000 do  -- to run training enough times
      output = model1:forward(input):clone()
   end

   model1:evaluate()
   local output2 = model1:forward(input):clone()

   mytester:assertTensorEq(output, output2,  0.0002, "issue 129 err")

   -- test with rnn
   local normalize = nn.Sequential()
   normalize:add(nn.SpatialBatchNormalization(2))

   local model = nn.Sequential()
   model:add(nn.SplitTable(1))  -- since sequencer expects table as input
   model:add(nn.Sequencer(normalize))  -- wrapping batch-normalization in a sequencer
   model:add(nn.JoinTable(1))  -- since sequencer outputs table

   input:resize(1, 4, 2, 64, 64)  -- time_step X batch_size X channels X height X width

   model:training()

   local output
   for i=1, 1000 do  -- to run training enough times
      output = model:forward(input):clone()
   end

   mytester:assertTensorEq(model1:get(1).running_mean, model:get(2).module.sharedClones[1].modules[1].running_mean, 0.000001)
   mytester:assertTensorEq(model:get(2).module.sharedClones[1].modules[1].running_mean, model:get(2).module.modules[1].modules[1].running_mean, 0.0000001)

   model:evaluate()
   local output2 = model:forward(input):clone()

   mytester:assertTensorEq(output, output2,  0.0002, "issue 129 err")
end

function rnntest.encoderdecoder()
   torch.manualSeed(123)

   local opt = {}
   opt.learningRate = 0.1
   opt.hiddenSz = 2
   opt.vocabSz = 5
   opt.inputSeqLen = 3 -- length of the encoded sequence

   --[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
   local function forwardConnect(encLSTM, decLSTM)
      decLSTM:setHiddenState(0, encLSTM:getHiddenState(opt.inputSeqLen))
   end

   --[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
   local function backwardConnect(encLSTM, decLSTM)
      encLSTM:setGradHiddenState(opt.inputSeqLen, decLSTM:getGradHiddenState(0))
   end

   -- Encoder
   local enc = nn.Sequential()
   enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
   enc:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
   local encLSTM = nn.RecGRU(opt.hiddenSz, opt.hiddenSz)
   enc:add(nn.Sequencer(encLSTM))
   enc:add(nn.SelectTable(-1))

   -- Decoder
   local dec = nn.Sequential()
   dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
   dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
   local decLSTM = nn.RecGRU(opt.hiddenSz, opt.hiddenSz)
   dec:add(nn.Sequencer(decLSTM))
   dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
   dec:add(nn.Sequencer(nn.LogSoftMax()))

   local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   local encParams, encGradParams = enc:getParameters()
   local decParams, decGradParams = dec:getParameters()

   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Some example data (batchsize = 2)
   local encInSeq = torch.Tensor({{1,2,3},{3,2,1}})
   local decInSeq = torch.Tensor({{1,2,3,4},{4,3,2,1}})
   local decOutSeq = torch.Tensor({{2,3,4,1},{1,2,4,3}})
   decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM)
   local decOut = dec:forward(decInSeq)
   local Edec = criterion:forward(decOut, decOutSeq)

   -- Backward pass
   local gEdec = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gEdec)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.zeros(encOut:size())
   enc:backward(encInSeq, zeroTensor)

   local function numgradtest()
      -- Here, we do a numerical gradient check to make sure the coupling is correct:
      local eps = 1e-5

      local decGP_est, encGP_est = torch.DoubleTensor(decGradParams:size()), torch.DoubleTensor(encGradParams:size())

      -- Easy function to do forward pass over coupled network and get error
      local function forwardPass()
         local encOut = enc:forward(encInSeq)
         forwardConnect(encLSTM, decLSTM)
         local decOut = dec:forward(decInSeq)
         local E = criterion:forward(decOut, decOutSeq)
         return E
      end

      -- Check encoder
      for i = 1, encGradParams:size(1) do
         -- Forward with \theta+eps
         encParams[i] = encParams[i] + eps
         local C1 = forwardPass()
         -- Forward with \theta-eps
         encParams[i] = encParams[i] - 2 * eps
         local C2 = forwardPass()

         encParams[i] = encParams[i] + eps
         encGP_est[i] = (C1 - C2) / (2 * eps)
      end
      mytester:assertTensorEq(encGradParams, encGP_est, eps, "Numerical gradient check for encoder failed")

      -- Check decoder
      for i = 1, decGradParams:size(1) do
         -- Forward with \theta+eps
         decParams[i] = decParams[i] + eps
         local C1 = forwardPass()
         -- Forward with \theta-eps
         decParams[i] = decParams[i] - 2 * eps
         local C2 = forwardPass()

         decParams[i] = decParams[i] + eps
         decGP_est[i] = (C1 - C2) / (2 * eps)
      end
      mytester:assertTensorEq(decGradParams, decGP_est, eps, "Numerical gradient check for decoder failed")
   end

   numgradtest()

   encGradParams:zero()
   decGradParams:zero()

   -- issue 142

   -- batchsize = 3

   encInSeq = torch.Tensor({{1,2,3},{3,2,1},{1,3,5}})
   decInSeq = torch.Tensor({{1,2,3,4},{4,3,2,1},{1,3,5,1}})
   decOutSeq = torch.Tensor({{2,3,4,1},{1,2,4,3},{1,2,5,3}})
   decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM)
   local decOut = dec:forward(decInSeq)
   local Edec = criterion:forward(decOut, decOutSeq)

   -- Backward pass
   local gEdec = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gEdec)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.zeros(encOut:size())
   enc:backward(encInSeq, zeroTensor)

   numgradtest()
end

function rnntest.reinforce()
   -- test that AbstractRecurrent:reinforce(rewards) words
   local seqLen = 4
   local batchSize = 3
   local rewards = {}
   for i=1,seqLen do
      rewards[i] = torch.randn(batchSize)
   end
   local rf = nn.ReinforceNormal(0.1)
   local rnn = nn.Recursor(rf)
   local input = torch.randn(batchSize,3)
   for i=1,seqLen do
      rnn:forward(input)
   end
   rnn:reinforce(rewards)
   for i=1,seqLen do
      local rm = rnn:getStepModule(i)
      mytester:assertTensorEq(rm.reward, rewards[i], 0.000001, "Reinforce error")
   end
end

function rnntest.rnnlm()
   if not pcall(function() require 'nngraph' end) then
      return
   end

   local vocabsize = 100
   local opt = {
      seqlen = 5,
      batchsize = 3,
      hiddensize = {20,20},
      lstm = true
   }

   local lm = nn.Sequential()

   -- input layer (i.e. word embedding space)
   local lookup = nn.LookupTable(vocabsize, opt.hiddensize[1])
   lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
   lm:add(lookup) -- input is seqlen x batchsize
   lm:add(nn.SplitTable(1)) -- tensor to table of tensors

   -- rnn layers
   local stepmodule = nn.Sequential() -- applied at each time-step
   local inputsize = opt.hiddensize[1]
   local rnns = {}
   for i,hiddensize in ipairs(opt.hiddensize) do
      nn.FastLSTM.usenngraph = true -- faster
      local rnn = nn.FastLSTM(inputsize, hiddensize)
      table.insert(rnns, rnn)
      stepmodule:add(rnn)
      inputsize = hiddensize
   end
   nn.FastLSTM.usenngraph = false
   -- output layer
   local linear = nn.Linear(inputsize, vocabsize)
   stepmodule:add(linear)
   stepmodule:add(nn.LogSoftMax())
   lm:add(nn.Sequencer(stepmodule))
   lm:remember('both')


   --[[ multiple sequencer ]]--


   local lm2 = nn.Sequential()

   local inputSize = opt.hiddensize[1]
   for i,hiddenSize in ipairs(opt.hiddensize) do
      local rnn = nn.Sequencer(rnns[i]:clone())
      lm2:add(rnn)
      inputSize = hiddenSize
   end

   -- input layer (i.e. word embedding space)
   lm2:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors
   local lookup2 = lookup:clone()
   lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
   lm2:insert(lookup2, 1)

   -- output layer
   local softmax = nn.Sequential()
   softmax:add(linear:clone())
   softmax:add(nn.LogSoftMax())
   lm2:add(nn.Sequencer(softmax))
   lm2:remember('both')

   -- compare

   for j=1,2 do
      local inputs = torch.LongTensor(opt.seqlen, opt.batchsize):random(1,vocabsize)
      local gradOutputs = torch.randn(opt.seqlen, opt.batchsize, vocabsize)
      local gradOutputs = nn.SplitTable(1):forward(gradOutputs)

      local params, gradParams = lm:parameters()
      local params2, gradParams2 = lm2:parameters()

      lm:training()
      lm2:training()
      for i=1,4 do
         local outputs = lm:forward(inputs)
         lm:zeroGradParameters()
         local gradInputs = lm:backward(inputs, gradOutputs)
         lm:updateParameters(0.1)

         local inputs2 = inputs:transpose(1,2)
         local outputs2 = lm2:forward(inputs2)
         lm2:zeroGradParameters()
         local gradInputs2 = lm2:backward(inputs2, gradOutputs)
         lm2:updateParameters(0.1)

         mytester:assertTensorEq(gradInputs, gradInputs2:transpose(1,2), 0.0000001, "gradInputs err")
         for k=1,#outputs2 do
            mytester:assertTensorEq(outputs2[k], outputs[k], 0.0000001, "outputs err "..k)
         end

         for k=1,#params do
            mytester:assertTensorEq(gradParams[k], gradParams2[k], 0.0000001, "gradParam err "..k)
            mytester:assertTensorEq(params[k], params2[k], 0.0000001, "param err"..k)
         end
      end

      lm:evaluate()
      lm2:evaluate()
      for i=1,3 do
         local outputs = lm:forward(inputs)

         local inputs2 = inputs:transpose(1,2)
         local outputs2 = lm2:forward(inputs2)

         for k=1,#outputs2 do
            mytester:assertTensorEq(outputs2[k], outputs[k], 0.0000001, "outputs err "..k)
         end
      end
   end
end

function rnntest.issue204()
   if not pcall(function() require 'optim' end) then
      return
   end

   -- Hyperparameters
   local inputSize = 3
   local hiddenSize = 2
   local nClasses = 4
   local nIndex = 10
   local maxSeqLen = 20
   local nSamples = 50
   local nEpochs = 10

   -- Creating dummy dataset
   local sentences = {}
   local targets = {}
   local i = 1
   for seqLen=4,5 do
     local seq = torch.Tensor(seqLen, inputSize):uniform(0,1)
     local target = torch.random(nClasses)
     sentences[i] = seq
     targets[i] = target
     i = i + 1
   end

   local sentences2 = {sentences[2]:clone(), sentences[1]:clone()}
   local targets2 = {targets[2], targets[1]}

   -- Defining model
   local sequencer = nn.Sequencer(nn.Linear(inputSize, hiddenSize))
   local rnn = nn.Sequential()
     :add(nn.SplitTable(1,2))
     :add(sequencer) --nn.FastLSTM(inputSize, hiddenSize)))
     :add(nn.SelectTable(-1))
     :add(nn.Linear(hiddenSize, nClasses))
     :add(nn.LogSoftMax())
   local criterion = nn.ClassNLLCriterion()
   local params, gradParams = rnn:getParameters()

   local rnn2 = rnn:clone()
   local criterion2 = criterion:clone()
   local params2, gradParams2 = rnn2:getParameters()

   -- problem occurs when sequence length is increased
   rnn2:zeroGradParameters()
   rnn:zeroGradParameters()

   local outputs, loss, gradOutputs, gradInputs = {}, {}, {}, {}
   local outputs2, loss2, gradOutputs2, gradInputs2 = {}, {}, {}, {}
   for i=1,2 do
      outputs[i] = rnn:forward(sentences[i]):clone()
      loss[i] = criterion:forward(outputs[i], targets[i])
      gradOutputs[i] = criterion:backward(outputs[i], targets[i]):clone()
      gradInputs[i] = rnn:backward(sentences[i], gradOutputs[i]):clone()

      outputs2[i] = rnn2:forward(sentences2[i]):clone()
      loss2[i] = criterion2:forward(outputs2[i], targets2[i])
      gradOutputs2[i] = criterion2:backward(outputs2[i], targets2[i]):clone()
      gradInputs2[i] = rnn2:backward(sentences2[i], gradOutputs2[i]):clone()

   end

   mytester:assertTensorEq(gradParams, gradParams2, 0.000001)
   mytester:assertTensorEq(outputs[1], outputs2[2], 0.000001)
   mytester:assertTensorEq(outputs[2], outputs2[1], 0.000001)
   mytester:assertTensorEq(gradInputs[1], gradInputs2[2], 0.000001)
   mytester:assertTensorEq(gradInputs[2], gradInputs2[1], 0.000001)
end

function rnntest.SeqLSTM_main()
   local inputsize = 2
   local outputsize = 3

   -- compare SeqLSTM to RecLSTM (forward, backward, update)
   local function testmodule(seqlstm, seqlen, batchsize, lstm2, remember, eval, seqlstm2, maskzero)

      lstm2 = lstm2 or seqlstm:toRecLSTM()
      remember = remember or 'neither'
      seqlstm2 = seqlstm2 or nn.Sequencer(lstm2)

      local input, gradOutput
      input = torch.randn(seqlen, batchsize, inputsize)
      if maskzero then
         lstm2:maskZero()
         local zeroMask = torch.ByteTensor(seqlen, batchsize):zero()
         for i=1,seqlen do
            for j=1,batchsize do
               if math.random() < 0.2 then
                  zeroMask[{i,j}] = 1
               end
            end
         end
         seqlstm:setZeroMask(zeroMask)
         seqlstm2:setZeroMask(zeroMask)
      else
          seqlstm:setZeroMask(false)
         seqlstm2:setZeroMask(false)
      end
      gradOutput = torch.randn(seqlen, batchsize, outputsize)

      seqlstm2:remember(remember)
      mytester:assert(seqlstm2._remember == remember, tostring(seqlstm2._remember) ..'~='.. tostring(remember))
      seqlstm:remember(remember)

      if eval then
         seqlstm:evaluate()
         seqlstm2:evaluate()
      else
         seqlstm:training()
         seqlstm2:training()
      end

      -- forward

      local output = seqlstm:forward(input)

      local output2 = seqlstm2:forward(input)
      mytester:assertTensorEq(output, output2, 0.000001)

      mytester:assertTableEq(output:size():totable(), gradOutput:size():totable(), 0.000001)

      if not eval then
         -- backward

         seqlstm:zeroGradParameters()
         seqlstm2:zeroGradParameters()
         local gradInput = seqlstm:backward(input, gradOutput)
         local gradInput2 = seqlstm2:backward(input, gradOutput)
         mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

         local params, gradParams = seqlstm:parameters()
         local params2, gradParams2 = lstm2:parameters()

         for i=1,#params do
            mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, tostring(gradParams2[i]:size()))
         end
      end

      return lstm2, seqlstm2
   end

   local seqlen = 4
   local batchsize = 5

   local seqlstm = nn.SeqLSTM(inputsize, outputsize):maskZero()
   seqlstm:reset(0.1)

   local lstm2 = testmodule(seqlstm, seqlen, batchsize)

   -- test forget

   local lstm2, seqlstm2 = testmodule(seqlstm, seqlen, batchsize, lstm2)

   -- test remember

   testmodule(seqlstm, seqlen, batchsize, lstm2, 'both', false, seqlstm2)
   mytester:assert(seqlstm._remember == 'both')

   -- test variable input size :

   local seqlen = 5
   local batchsize = 6

   testmodule(seqlstm, seqlen, batchsize)

   -- test forget (eval)

   local eval = true

   local p1 = seqlstm:getParameters()
   local p2 = lstm2:getParameters()
   mytester:assertTensorEq(p1, p2, 0.0000001)
   testmodule(seqlstm, seqlen, batchsize, lstm2, nil, eval, seqlstm2) --
   mytester:assert(seqlstm._remember == 'neither')

   -- test remember (eval)

   local p1 = seqlstm:getParameters()
   local p2 = lstm2:getParameters()
   mytester:assertTensorEq(p1, p2, 0.0000001)
   testmodule(seqlstm, seqlen, batchsize, lstm2, 'both', eval, seqlstm2) --
   mytester:assert(seqlstm.train == false)
   mytester:assert(lstm2.train == false)
   mytester:assert(seqlstm._remember == 'both')

   -- test variable input size (eval) :

   local seqlen = 4
   local batchsize = 5

   testmodule(seqlstm, seqlen, batchsize, lstm2, nil, eval)

   -- test variable length sequences

   seqlstm.maskzero = true
   lstm2:maskZero(1)

   testmodule(seqlstm, seqlen, batchsize, lstm2, nil, false, nil, true)
end

function rnntest.SeqLSTM_Lua_vs_C()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5

   local input = torch.randn(seqlen, batchsize, inputsize)

   local seqlstm = nn.SeqLSTM(inputsize, outputsize)
   local seqlstm2 = nn.SeqLSTM(inputsize, outputsize)
   seqlstm2.weight:copy(seqlstm.weight)
   seqlstm2.bias:copy(seqlstm.bias)
   seqlstm2.forceLua = true

   local output = seqlstm:forward(input)
   local output2 = seqlstm2:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   seqlstm:zeroGradParameters()
   seqlstm2:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)

   local gradInput = seqlstm:backward(input, gradOutput)
   local gradInput2 = seqlstm2:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqlstm:parameters()
   local params2, gradParams2 = seqlstm2:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.SeqLSTM_maskzero()
   -- tests that it works with non-masked inputs regardless of maskzero's value.
   -- Note that more maskzero = true tests with masked inputs are in SeqLSTM unit test.
   local T, N, D, H = 3, 2, 4, 5
   local seqlstm = nn.SeqLSTM(D,H)
   local seqlstm2 = seqlstm:clone():maskZero()
   local zeroMask = torch.ByteTensor(T, N):zero()
   seqlstm2:setZeroMask(zeroMask)

   local input = torch.randn(T, N, D)
   local gradOutput = torch.randn(T, N, H)

   local output = seqlstm:forward(input)
   local output2 = seqlstm2:forward(input)

   mytester:assertTensorEq(output, output2, 0.000001)

   seqlstm:zeroGradParameters()
   local gradInput = seqlstm:backward(input, gradOutput)
   seqlstm2:zeroGradParameters()
   local gradInput2 = seqlstm2:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqlstm:getParameters()
   local params2, gradParams2 = seqlstm2:getParameters()

   mytester:assertTensorEq(gradParams, gradParams2, 0.000001)
   if benchmark then
      local T, N, D, H = 20, 20, 50, 50
      if pcall(function() require 'cunn' end) then
         T, N, D, H = 100, 128, 250, 250
      end

      local seqlstm = nn.SeqLSTM(D,H)
      local input = torch.randn(T, N, D)
      local gradOutput = torch.randn(T, N, H)
      local zeroMask = torch.ByteTensor(T, N):random(0,1)

      if cunn then
         input = input:cuda()
         gradOutput = gradOutput:cuda()
         seqlstm:cuda()
         zeroMask = zeroMask:cudaByte()
      end

      seqlstm:forward(input)
      seqlstm:backward(input, gradOutput)

      if cunn then cutorch.synchronize() end
      local a = torch.Timer()
      for i=1,5 do
         seqlstm:forward(input)
         seqlstm:backward(input, gradOutput)
      end
      if cunn then cutorch.synchronize() end
      local nonmasktime = a:time().real

      seqlstm:maskZero()
      seqlstm:setZeroMask(zeroMask)
      seqlstm:forward(input)
      seqlstm:backward(input, gradOutput)

      if cunn then cutorch.synchronize() end
      local a = torch.Timer()
      for i=1,5 do
         seqlstm:forward(input)
         seqlstm:backward(input, gradOutput)
      end
      if cunn then cutorch.synchronize() end
      local masktime = a:time().real
      print("mask vs nonmask SeqLSTM", masktime, nonmasktime)
   end
end

function rnntest.SeqLSTMP_main()
   -- test that LSTM = LSTMP when projection layer is identity
   local inputsize = 2
   local hiddensize = 3
   local outputsize = 3
   local seqlen = 6
   local batchsize = 5

   local lstm = nn.SeqLSTM(inputsize, outputsize)
   local lstmp = nn.SeqLSTM(inputsize, hiddensize, outputsize)

   local params, gradParams = lstm:parameters()
   local paramsp, gradParamsp = lstmp:parameters()

   mytester:assert(#params + 1 == #paramsp)

   for i=1,#params do
      paramsp[i]:copy(params[i])
   end

   local wO = paramsp[3]
   mytester:assertTableEq(wO:size():totable(), {3,3})
   wO:eye(3,3)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local gradOutput = torch.randn(seqlen, batchsize, outputsize)

   local output = lstm:forward(input)
   local outputp = lstmp:forward(input)

   mytester:assertTensorEq(output, outputp, 0.000001)

   lstm:zeroGradParameters()
   lstmp:zeroGradParameters()

   mytester:assert(math.abs(gradParamsp[3]:sum()) < 0.00001)

   local gradInput = lstm:backward(input, gradOutput)
   local gradInputp = lstmp:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInputp, 0.000001)

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParamsp[i], 0.000001)
   end

   mytester:assert(math.abs(gradParamsp[3]:sum()) > 0.00001)

   -- test with maskzero

   lstmp:maskZero()
   lstm:maskZero()

   local zeroMask = torch.ByteTensor(seqlen, batchsize):zero()

   for i=1,seqlen do
      for j=1,batchsize do
         if math.random() < 0.2 then
            zeroMask[{i,j}] = 1
         end
      end
   end

   lstmp:setZeroMask(zeroMask)
   lstm:setZeroMask(zeroMask)

   local output = lstm:forward(input)
   local outputp = lstmp:forward(input)

   mytester:assertTensorEq(output, outputp, 0.000001)

   lstm:zeroGradParameters()
   lstmp:zeroGradParameters()

   local gradInput = lstm:backward(input, gradOutput)
   local gradInputp = lstmp:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInputp, 0.000001)

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParamsp[i], 0.000001)
   end

   mytester:assert(math.abs(gradParamsp[3]:sum()) > 0.00001)

   -- test with hiddensize ~= outputsize and maskzero
   lstm = nil

   local hiddensize = 4

   local lstmp = nn.SeqLSTM(inputsize, hiddensize, outputsize)
   local lstmp2 = nn.SeqLSTM(inputsize, hiddensize, outputsize)

   local params, gradParams = lstmp:parameters()
   local params2, gradParams2 = lstmp2:parameters()

   for i=1,#params do
      params[i]:copy(params2[i])
   end

   lstmp:zeroGradParameters()
   lstmp2:zeroGradParameters()

   local input = torch.randn(seqlen, batchsize, inputsize)
   zeroMask:zero()
   zeroMask[3] = 1 -- zero the 3rd time-step

   lstmp:maskZero()
   lstmp:setZeroMask(zeroMask)
   local output = lstmp:forward(input)
   local gradInput = lstmp:backward(input, gradOutput)

   lstmp2:remember('neither')
   local input1, input2 = input:sub(1,2), input:sub(4,seqlen)
   local gradOutput1, gradOutput2 = gradOutput:sub(1,2), gradOutput:sub(4,seqlen)
   local output2 = torch.zeros(output:size())
   local gradInput2 = torch.zeros(gradInput:size())
   output2:sub(1,2):copy(lstmp2:forward(input1))
   gradInput2:sub(1,2):copy(lstmp2:backward(input1, gradOutput1))
   output2:sub(4,seqlen):copy(lstmp2:forward(input2))
   gradInput2:sub(4,seqlen):copy(lstmp2:backward(input2, gradOutput2))

   mytester:assertTensorEq(output, output2, 0.000001)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, 'error in gradParams '..i)
   end
end

function rnntest.SeqBRNNTest()
   local brnn = nn.SeqBRNN(5, 5)

   local input = torch.rand(5, 1, 5)
   local output = brnn:forward(input)
   local concatTable = brnn.modules[1]:get(1)
   local fwd = concatTable:get(1) -- get SeqLSTM fwd.
   local bwd = concatTable:get(2):get(2) -- get SeqLSTM bwd.
   fwd:clearState()
   bwd:clearState()

   local fwdOutput = fwd:forward(input)

   local reverseSequence = nn.SeqReverseSequence(1)

   local reversedInput = reverseSequence:forward(input)
   local bwdOutput = bwd:forward(reversedInput)
   local bwdOutput = reverseSequence:forward(bwdOutput)

   local expectedOutput = torch.add(fwdOutput, bwdOutput)
   mytester:assertTensorEq(expectedOutput, output, 0)
end

function rnntest.SeqBRNNJoinTest()
   local brnn = nn.SeqBRNN(5, 5, false , nn.JoinTable(3))

   local input = torch.rand(5, 1, 5)
   local output = brnn:forward(input)
   local concatTable = brnn.modules[1]:get(1)
   local fwd = concatTable:get(1) -- get SeqLSTM fwd.
   local bwd = concatTable:get(2):get(2) -- get SeqLSTM bwd.
   fwd:clearState()
   bwd:clearState()

   local fwdOutput = fwd:forward(input)

   local reverseSequence = nn.SeqReverseSequence(1)

   local reversedInput = reverseSequence:forward(input)
   local bwdOutput = bwd:forward(reversedInput)
   local bwdOutput = reverseSequence:forward(bwdOutput)

   local expectedOutput = nn.JoinTable(3):forward({fwdOutput, bwdOutput})
   mytester:assertTensorEq(expectedOutput, output, 0)
end

function rnntest.BRNNBatchFirstTest()
   local brnn = nn.SeqBRNN(5, 5, true , nn.JoinTable(3))

   local input = torch.rand(1, 5, 5)
   local output = brnn:forward(input)
   local concatTable = brnn.modules[1]:get(2)
   local fwd = concatTable:get(1) -- get SeqLSTM fwd.
   local bwd = concatTable:get(2):get(2) -- get SeqLSTM bwd.
   fwd:clearState()
   bwd:clearState()

   input = input:transpose(1,2) -- Manually transpose the input.
   local fwdOutput = fwd:forward(input)

   local reverseSequence = nn.SeqReverseSequence(1)

   local reversedInput = reverseSequence:forward(input)
   local bwdOutput = bwd:forward(reversedInput)
   local bwdOutput = reverseSequence:forward(bwdOutput)

   local expectedOutput = nn.JoinTable(3):forward({fwdOutput, bwdOutput})
   local expectedOutput = expectedOutput:transpose(1,2) -- Undo transpose to input.
   mytester:assertTensorEq(expectedOutput, output, 0)
end

function rnntest.clearState()
   local seq = nn.Sequential()
   local seqLSTM = nn.LSTM(200, 4)
   seq:add(nn.Sequencer(seqLSTM))

   for i=1,10 do
      seq:forward({torch.Tensor(200), torch.Tensor(200), torch.Tensor(200)})
   end
   local criterion = nn.SequencerCriterion(nn.MSECriterion())
   for i=1,10 do
      local input = {torch.Tensor(200), torch.Tensor(200), torch.Tensor(200)}
      local t = {torch.Tensor(4), torch.Tensor(4), torch.Tensor(4)}
      local output = seq:forward(input)
   end

   local nsharedclone = #seqLSTM.sharedClones
   seq:clearState()

   -- Test if shared clones are deleted
   mytester:assert(#seqLSTM.sharedClones == nsharedclone, 'sharedClones should remain after clear')
   mytester:assert(#seqLSTM.cells == 0, 'cells should be empty after clear')
   mytester:assert(#seqLSTM.gradCells == 0, 'gradCells should be empty after clear')
   mytester:assert(seqLSTM.nSharedClone == nsharedclone, 'nSharedClone should reflect count')

   for i=1,nsharedclone do
      mytester:assert(#seqLSTM.sharedClones[i].output == 0, 'shared clones should be cleared of state')
   end

   -- Make sure it still works after clearing
   for i=1,10 do
      local input = {torch.Tensor(200), torch.Tensor(200), torch.Tensor(200)}
      local t = {torch.Tensor(4), torch.Tensor(4), torch.Tensor(4)}
      local output = seq:forward(input)
   end
end

function rnntest.NormStabilizer()
   if not pcall(function() require "optim" end) then
      return
   end
   local SequencerCriterion, parent = torch.class('nn.SequencerCriterionNormStab', 'nn.SequencerCriterion')

   function SequencerCriterion:__init(criterion, beta)
      parent.__init(self, criterion)
      self.beta = beta
   end

   function SequencerCriterion:updateOutput(inputTable, targetTable)
      self.output = 0
      for i,input in ipairs(inputTable) do
         local criterion = self:getStepCriterion(i)
         self.output = self.output + criterion:forward(input, targetTable[i])
         if i > 1 then
            local reg = 0
            for j=1,input:size(1) do
               reg = reg + ((input[j]:norm() - inputTable[i-1][j]:norm())^2)
            end
            self.output = self.output + self.beta * reg / input:size(1)
         end
      end
      return self.output
   end

   -- Make a simple RNN and training set to test gradients
   -- hyper-parameters
   local batchSize = 3
   local seqlen = 2
   local hiddenSize = 3
   local inputSize = 4
   local lr = 0.1
   local beta = 50.0

   local r = nn.LinearRNN(inputSize, hiddenSize)

   -- build simple recurrent neural network
   local rnn = nn.Sequential()
      :add(r)
      :add(nn.NormStabilizer(beta))

   rnn = nn.Sequencer(rnn)
   local criterion = nn.SequencerCriterionNormStab(nn.MSECriterion(), beta)

   local iteration = 1
   local params, gradParams = rnn:getParameters()

   while iteration < 5 do
      -- generate a random data point
      local inputs, targets = {}, {}
      for step=1,seqlen do
         inputs[step] = torch.randn(batchSize, inputSize)
         targets[step] = torch.randn(batchSize, hiddenSize)
      end

      -- set up closure
      local function feval(params_new)
         if params ~= params_new then
            params:copy(params_new)
         end

         rnn:zeroGradParameters()
         local outputs = rnn:forward(inputs)
         local err = criterion:forward(outputs, targets)
         local gradOutputs = criterion:backward(outputs, targets)
         local gradInputs = rnn:backward(inputs, gradOutputs)
         return err, gradParams
      end

      -- compare numerical to analytic gradient
      local diff, dC, dC_est = optim.checkgrad(feval, params, 1e-10)
      mytester:assert(diff < 1e-3, "Numerical gradient and analytic gradient do not match.")

      rnn:updateParameters(lr)

      iteration = iteration + 1
   end

   -- compare to other implementation :
   local NS, parent = torch.class("nn.NormStabilizerTest", "nn.AbstractRecurrent")

   function NS:__init(beta)
      parent.__init(self, nn.CopyGrad())
      self.beta = beta
   end

   function NS:_accGradParameters(input, gradOutput, scale)
      -- No parameters to update
   end

   function NS:updateOutput(input)
      local output
      if self.train ~= false then
         self:recycle()
         local recurrentModule = self:getStepModule(self.step)
         output = recurrentModule:updateOutput(input)
      else
         output = self.recurrentModule:updateOutput(input)
      end

      self.outputs[self.step] = output

      self.output = output
      self.step = self.step + 1
      self.gradPrevOutput = nil
      self.updateGradInputStep = nil
      self.accGradParametersStep = nil

      return self.output
   end

   function NS:_updateGradInput(input, gradOutput)
      -- First grab h[t] and h[t+1] :
      -- backward propagate through this step
      local curStep = self.updateGradInputStep-1
      local hiddenModule = self:getStepModule(curStep)
      hiddenModule:updateGradInput(input, gradOutput)
      local hiddenState = hiddenModule.output

      if curStep < self.step then
         local batchSize = hiddenState:size(1)
         if curStep > 1 then
            local prevHiddenModule = self:getStepModule(curStep - 1)
            local prevHiddenState = prevHiddenModule.output
            -- Add norm stabilizer cost function directly to respective CopyGrad.gradInput tensors
            for i=1,batchSize do
               local dRegdNorm =  self.beta * 2 * (hiddenState[i]:norm()-prevHiddenState[i]:norm()) / batchSize
               local dNormdHid = torch.div(hiddenState[i], hiddenState[i]:norm())
               hiddenModule.gradInput[i]:add(torch.mul(dNormdHid, dRegdNorm))
            end
         end
         if curStep < self.step-1 then
            local nextHiddenModule = self:getStepModule(curStep + 1)
            local nextHiddenState = nextHiddenModule.output
            for i=1,batchSize do
               local dRegdNorm = self.beta * -2 * (nextHiddenState[i]:norm() - hiddenState[i]:norm()) / batchSize
               local dNormdHid = torch.div(hiddenState[i], hiddenState[i]:norm())
               hiddenModule.gradInput[i]:add(torch.mul(dNormdHid, dRegdNorm))
            end
         end
      end
      return hiddenModule.gradInput
   end

   local ns = nn.NormStabilizer(beta)
   local ns2 = nn.NormStabilizerTest(beta)

   local seq = nn.Sequencer(ns)
   local seq2 = nn.Sequencer(ns2)

   local inputs, gradOutputs = {}, {}
   for step=1,seqlen do
      inputs[step] = torch.randn(batchSize, inputSize)
      gradOutputs[step] = torch.randn(batchSize, inputSize)
   end

   local outputs = seq:forward(inputs)
   local outputs2 = seq2:forward(inputs)
   local gradInputs = seq:backward(inputs, gradOutputs)
   local gradInputs2 = seq2:backward(inputs, gradOutputs)

   for step=1,seqlen do
      mytester:assertTensorEq(outputs[step], outputs2[step], 0.0000001)
      mytester:assertTensorEq(gradInputs[step], gradInputs2[step], 0.0000001)
   end

   ns:updateLoss()
end

function rnntest.NCE_MaskZero()
   local opt = {
      datasize = 20,
      batchsize = 4,
      seqlen = 5,
      uniform = 0.1,
      hiddensize = {100},
      vocabsize = 100,
      k = 25
   }

   local lm = nn.Sequential()

   -- input layer (i.e. word embedding space)
   local lookup = nn.LookupTableMaskZero(opt.vocabsize, opt.hiddensize[1])
   lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
   lm:add(lookup) -- input is seqlen x batchsize

   -- rnn layers
   local inputsize = opt.hiddensize[1]
   for i,hiddensize in ipairs(opt.hiddensize) do
      lm:add(nn.SeqLSTM(inputsize, hiddensize):maskZero())
      lm:add(nn.Dropout(opt.dropout))
      inputsize = hiddensize
   end

   lm:add(nn.SplitTable(1))

   -- output layer
   local unigram = torch.FloatTensor():range(1,opt.vocabsize)
   unigram:pow(2)
   local ncemodule = nn.NCEModule(inputsize, opt.vocabsize, opt.k, unigram)

   -- NCE requires {input, target} as inputs
   lm = nn.Sequential()
      :add(nn.ParallelTable()
         :add(lm):add(nn.Identity()))
      :add(nn.ZipTable()) -- {{x1,x2,...}, {t1,t2,...}} -> {{x1,t1},{x2,t2},...}

   -- encapsulate stepmodule into a Sequencer
   lm:add(nn.Sequencer(nn.MaskZero(ncemodule)))

   -- remember previous state between batches
   lm:remember()

   if opt.uniform > 0 then
      for k,param in ipairs(lm:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
      ncemodule:reset()
   end

   --[[ loss function ]]--

   local crit = nn.MaskZeroCriterion(nn.NCECriterion())

   local targetmodule =  nn.SplitTable(1)
   local criterion = nn.SequencerCriterion(crit)

   local data = {
      inputs = torch.LongTensor(opt.datasize, opt.seqlen, opt.batchsize):random(0,opt.vocabsize),
      targets = torch.LongTensor(opt.datasize, opt.seqlen, opt.batchsize):random(1,opt.vocabsize)
   }

   local _ = require 'moses'
   local starterr
   local err
   local found = false
   for epoch=1,12 do
      err = 0
      for i=1,opt.datasize do
         local input, target = data.inputs[i], data.targets[i]
         local target = targetmodule:forward(target)
         local zeroMask = nn.utils.getZeroMaskSequence(input)
         --print("INPUT ZM", input, zeroMask)
         lm:setZeroMask(zeroMask)
         criterion:setZeroMask(zeroMask)
         local output = lm:forward({input, target})
         err = err + criterion:forward(output, target)
         local gradOutput = criterion:backward(output, target)
         if not found then
            for i=1,input:size(1) do
               for j=1,input:size(2) do
                  if input[{i,j}] == 0 then
                     found = true
                     -- test that it works with mask zero
                     mytester:assert(output[i][1][j] == 0)
                     mytester:assert(gradOutput[i][1][j] == 0)
                  end
               end
            end
         end
         lm:zeroGradParameters()
         local gradInput = lm:backward({input, target}, gradOutput)
         lm:updateParameters(0.05)
      end
      if epoch == 1 then
         starterr = err
      end
      if epoch > 3 and err < starterr then
         break
      end
   end
   mytester:assert(found)
   mytester:assert(err < starterr, string.format("err=%f should be smaller than starterr=%f", err, starterr))
end

local function check_size(x, dims)
  mytester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    mytester:assert(x:size(i) == d)
  end
end

-- Make sure that everything works when we don't pass initial hidden or initial
-- cell state; in this case we only pass input sequence of vectors
function rnntest.SeqGRU_noHiddenTest()
  local N, T, D, H = 4, 5, 6, 7
  local gru = nn.SeqGRU(D, H)

  for t = 1, 3 do
    local x = torch.randn(T, N, D)
    local dout = torch.randn(T, N, H)

    local out = gru:forward(x)
    local din = gru:backward(x, dout)

    mytester:assert(torch.isTensor(din))
    check_size(din, {T, N, D})

    -- Make sure the initial hidden state are zero
    mytester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
  end
end


function rnntest.SeqGRU_rememberStatesTest()
  local N, T, D, H = 5, 6, 7, 8
  local gru = nn.SeqGRU(D, H)
  gru:remember('both')

  local final_h = nil
  for t = 1, 4 do
    local x = torch.randn(T, N, D)
    local dout = torch.randn(T, N, H)
    local out = gru:forward(x)
    local din = gru:backward(x, dout)

    if t == 1 then
      mytester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
    elseif t > 1 then
      mytester:assertTensorEq(gru.h0, final_h, 0)
    end
    final_h = out[T]:clone()
  end

  -- Initial states should reset to zero after we call resetStates
  gru:forget()
  local x = torch.randn(T, N, D)
  local dout = torch.randn(T, N, H)
  gru:forward(x)
  gru:backward(x, dout)
  mytester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
end

function rnntest.SeqGRU_main()
   local inputsize, outputsize = 2, 3
   -- compare SeqGRU to GRU (forward, backward, update)
   local function testmodule(seqGRU, seqlen, batchsize, gru2, remember, eval, seqGRU2, maskzero)

      gru2 = gru2 or seqGRU:toRecGRU()
      remember = remember or 'neither'
      seqGRU2 = seqGRU2 or nn.Sequencer(gru2)

      local input = torch.randn(seqlen, batchsize, inputsize)
      if maskzero then
         seqGRU:maskZero()
         gru2:maskZero()
         local zeroMask = torch.ByteTensor(seqlen, batchsize):zero()
         for i=1,seqlen do
            for j=1,batchsize do
               if math.random() < 0.2 then
                  zeroMask[{i,j}] = 1
               end
            end
         end
         seqGRU:setZeroMask(zeroMask)
         seqGRU2:setZeroMask(zeroMask)
      else
         seqGRU:setZeroMask(false)
         seqGRU2:setZeroMask(false)
      end
      local gradOutput = torch.randn(seqlen, batchsize, outputsize)

      seqGRU2:remember(remember)
      mytester:assert(seqGRU2._remember == remember, tostring(seqGRU2._remember) ..'~='.. tostring(remember))
      seqGRU:remember(remember)

      if eval then
         seqGRU:evaluate()
         seqGRU2:evaluate()
      else
         seqGRU:training()
         seqGRU2:training()
      end

      -- forward

      local output = seqGRU:forward(input)
      local output2 = seqGRU2:forward(input)
      mytester:assertTensorEq(output, output2, 0.000001)
      mytester:assertTableEq(output:size():totable(), gradOutput:size():totable(), 0.000001)

      if not eval then
         -- backward

         seqGRU:zeroGradParameters()
         seqGRU2:zeroGradParameters()
         local gradInput = seqGRU:backward(input, gradOutput)
         local gradInput2 = seqGRU2:backward(input, gradOutput)
         mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

         local params, gradParams = seqGRU:parameters()
         local params2, gradParams2 = gru2:parameters()

         for i=1,#params do
            mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, tostring(gradParams2[i]:size()))
         end
      end

      return gru2, seqGRU2
   end

   local seqlen = 5
   local batchsize = 6

   local seqGRU = nn.SeqGRU(inputsize, outputsize)
   seqGRU:reset(0.1)

   local gru2 = testmodule(seqGRU, seqlen, batchsize)

   -- test forget

   local gru2, seqGRU2 = testmodule(seqGRU, seqlen, batchsize, gru2)

   -- test remember

   testmodule(seqGRU, seqlen, batchsize, gru2, 'both', false, seqGRU2)
   mytester:assert(seqGRU._remember == 'both')

   -- test variable input size :

   local seqlen = 4
   local batchsize = 5

   testmodule(seqGRU, seqlen, batchsize)

   -- test forget (eval)

   local eval = true

   local p1 = seqGRU:getParameters()
   local p2 = gru2:getParameters()
   mytester:assertTensorEq(p1, p2, 0.0000001)
   testmodule(seqGRU, seqlen, batchsize, gru2, nil, eval, seqGRU2)
   mytester:assert(seqGRU._remember == 'neither')

   -- test remember (eval)

   local p1 = seqGRU:getParameters()
   local p2 = gru2:getParameters()
   mytester:assertTensorEq(p1, p2, 0.0000001)
   testmodule(seqGRU, seqlen, batchsize, gru2, 'both', eval, seqGRU2)
   mytester:assert(seqGRU.train == false)
   mytester:assert(gru2.train == false)
   mytester:assert(seqGRU._remember == 'both')

   -- test variable input size (eval) :

   local seqlen = 4
   local batchsize = 5

   testmodule(seqGRU, seqlen, batchsize, gru2, nil, eval)

   -- test variable length sequences

   testmodule(seqGRU, seqlen, batchsize, gru2, nil, false, nil, true)
end

function rnntest.SeqGRU_maskzero()
   -- tests that it works with non-masked inputs regardless of maskzero's value..
   local T, N, D, H = 3, 2, 4, 5
   local seqGRU = nn.SeqGRU(D,H) -- maskzero - false
   local seqGRU2 = seqGRU:clone():maskZero()
   local zeroMask = torch.ByteTensor(T, N):zero()
   seqGRU2:setZeroMask(zeroMask)

   local input = torch.randn(T, N, D)
   local gradOutput = torch.randn(T, N, H)

   local output = seqGRU:forward(input)
   local output2 = seqGRU2:forward(input)

   mytester:assertTensorEq(output, output2, 0.000001)

   seqGRU:zeroGradParameters()
   local gradInput = seqGRU:backward(input, gradOutput)
   seqGRU2:zeroGradParameters()
   local gradInput2 = seqGRU2:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqGRU:getParameters()
   local params2, gradParams2 = seqGRU2:getParameters()

   mytester:assertTensorEq(gradParams, gradParams2, 0.000001)
   if benchmark then
      local T, N, D, H = 20, 20, 50, 50
      if pcall(function() require 'cunn' end) then
         T, N, D, H = 100, 128, 250, 250
      end

      local seqGRU = nn.SeqGRU(D,H)
      local input = torch.randn(T, N, D)
      local gradOutput = torch.randn(T, N, H)
      local zeroMask = torch.ByteTensor(T, N):random(0,1)

      if cunn then
         input = input:cuda()
         gradOutput = gradOutput:cuda()
         seqGRU:cuda()
         zeroMask = zeroMask:cudaByte()
      end

      seqGRU:forward(input)
      seqGRU:backward(input, gradOutput)

      if cunn then cutorch.synchronize() end
      local a = torch.Timer()
      for i=1,5 do
         seqGRU:forward(input)
         seqGRU:backward(input, gradOutput)
      end
      if cunn then cutorch.synchronize() end
      local nonmasktime = a:time().real

      seqGRU:maskZero()
      seqGRU:setZeroMask(zeroMask)
      seqGRU:forward(input)
      seqGRU:backward(input, gradOutput)

      if cunn then cutorch.synchronize() end
      local a = torch.Timer()
      for i=1,5 do
         seqGRU:forward(input)
         seqGRU:backward(input, gradOutput)
      end
      if cunn then cutorch.synchronize() end
      local masktime = a:time().real
      print("mask vs nonmask SeqGRU", masktime, nonmasktime)
   end
end

function rnntest.SeqGRU_Lua_vs_C()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5

   local input = torch.randn(seqlen, batchsize, inputsize)

   local seqgru = nn.SeqGRU(inputsize, outputsize)
   local seqgru2 = nn.SeqGRU(inputsize, outputsize)
   seqgru2.weight:copy(seqgru.weight)
   seqgru2.bias:copy(seqgru.bias)
   seqgru2.forceLua = true

   local output = seqgru:forward(input)
   local output2 = seqgru2:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   seqgru:zeroGradParameters()
   seqgru2:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)

   local gradInput = seqgru:backward(input, gradOutput)
   local gradInput2 = seqgru2:backward(input, gradOutput)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqgru:parameters()
   local params2, gradParams2 = seqgru2:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function checkgrad(opfunc, x, eps)
    -- compute true gradient:
    local _,dC = opfunc(x)
    dC:resize(x:size())
    dC = dC:clone()

    -- compute numeric approximations to gradient:
    local eps = eps or 1e-7
    local dC_est = torch.DoubleTensor(dC:size())
    for i = 1,dC:size(1) do
      x[i] = x[i] + eps
      local C1 = opfunc(x)
      x[i] = x[i] - 2 * eps
      local C2 = opfunc(x)
      x[i] = x[i] + eps
      dC_est[i] = (C1 - C2) / (2 * eps)
    end

    -- estimate error of gradient:
    local diff = torch.norm(dC - dC_est) / torch.norm(dC + dC_est)
    return diff,dC,dC_est
end

function rnntest.MufuruGradients()
   local batchSize = 2
   local inputSize = 3
   local outputSize = 4
   local seqlen = 5

   local rnn = nn.MuFuRu(inputSize, outputSize)
   local module = nn.Sequencer(rnn)
   local w,dw = module:getParameters()
   local crit = nn.SequencerCriterion(nn.CrossEntropyCriterion())

   local input = torch.randn(seqlen, batchSize, inputSize)
   local target = torch.LongTensor(seqlen, batchSize):random(1,outputSize)
   local function feval(x)
      if w ~= x then w:copy(x) end
      module:zeroGradParameters()
      local out = module:forward(input)
      local err = crit:forward(out, target)
      local gradOutput = crit:backward(out, target)
      module:backward(input, gradOutput)
      return err, dw
   end
   local err = checkgrad(feval, w:clone())
   mytester:assertlt(err, precision, "error in computing grad parameters")
end

function rnntest.getHiddenState()
   local seqlen, batchsize = 7, 3
   local inputsize, outputsize = 4, 5
   local input = torch.randn(seqlen*2, batchsize, inputsize)
   local gradOutput = torch.randn(seqlen*2, batchsize, outputsize)

   local function testHiddenState(lstm, recurrence)
      local lstm2 = lstm:clone()

      -- test forward
      for step=1,seqlen do -- initialize lstm2 hidden state
         lstm2:forward(input[step])
      end

      for step=1,seqlen do
         local hiddenState = lstm2:getHiddenState(seqlen+step-1)
         if torch.type(hiddenState) == 'table' then
            mytester:assert(#hiddenState >= 1)
         else
            mytester:assert(torch.isTensor(hiddenState))
         end
         lstm:setHiddenState(step-1, hiddenState)
         local output = lstm:forward(input[seqlen+step])
         local output2 = lstm2:forward(input[seqlen+step])
         mytester:assertTensorEq(output, output2, 0.0000001, "error in step "..step)
      end

      -- test backward
      lstm:zeroGradParameters()
      lstm2:zeroGradParameters()
      lstm:forget()
      lstm:setStartState() -- empties start states

      for step=1,seqlen do
         lstm:forward(input[step])
         local hs = lstm:getHiddenState(step)
         local hs2 = lstm2:getHiddenState(step)
         if torch.type(hs) == 'table' then
            if recurrence then
               hs = hs[1][1]
               hs2 = hs2[1][1]
            end
            if torch.type(hs) == 'table' then
               for i=1,#hs do
                  mytester:assertTensorEq(hs[i], hs2[i], 0.0000001)
               end
            else
               mytester:assertTensorEq(hs, hs2, 0.0000001)
            end
         else
            mytester:assertTensorEq(hs, hs2, 0.0000001)
         end
      end

      for step=seqlen*2,seqlen+1,-1 do
         lstm2:backward(input[step], gradOutput[step])
      end

      lstm2:zeroGradParameters()

      for step=seqlen,1,-1 do
         local gradHiddenState = lstm2:getGradHiddenState(step)
         if torch.type(gradHiddenState) == 'table' then
            mytester:assert(#gradHiddenState >= 1)
         else
            mytester:assert(torch.isTensor(gradHiddenState))
         end
         lstm:setGradHiddenState(step, gradHiddenState)
         local gradInput = lstm:backward(input[step], gradOutput[step])
         local gradInput2 = lstm2:backward(input[step], gradOutput[step])
         mytester:assertTensorEq(gradInput, gradInput2, 0.0000001)
      end

      local params, gradParams = lstm:parameters()
      local params2, gradParams2 = lstm2:parameters()

      for i=1,#params do
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00000001)
      end
   end

   local lstm = nn.RecLSTM(inputsize, outputsize)
   testHiddenState(lstm)

   local gru = nn.RecGRU(inputsize, outputsize)
   testHiddenState(gru)

   gru:forget()
   gru:setStartState()
   testHiddenState(nn.Recursor(gru), false)

   testHiddenState(nn.LinearRNN(inputsize, outputsize), true)
end


function rnntest.VariableLength_FromSamples()
   torch.manualSeed(0)
   local nSamples = 10
   local maxLength = 20
   for run=1,10 do
      local lengths = torch.LongTensor(nSamples)
      lengths:random(maxLength)
      local samples = {}
      for i=1,nSamples do
         local t = torch.rand(lengths[i], 5)
         samples[i] = t
      end
      local output = torch.Tensor()
      local mask = torch.ByteTensor()
      local indexes, mappedLengths = output.nn.VariableLength_FromSamples(samples, output, mask)

      for i, ids in ipairs(indexes) do
         local m = mask:select(2, i)
         local t = output:select(2, i)
         for j, sampleId in ipairs(ids) do
            local l = lengths[sampleId]
            -- check that the length was mapped correctly
            mytester:assert(l == mappedLengths[i][j])
            -- checks that the mask is 0 for valid entries
            mytester:assert(math.abs(m:narrow(1, 1, l):sum()) < 0.000001)
            -- checks that the valid entries are equal
            mytester:assertTensorEq(t:narrow(1, 1, l), samples[sampleId])
            if l < m:size(1) then
               mytester:assert(m[l+1] == 1)
            end
            if l+1 < m:size(1) then
               m = m:narrow(1, l+2, m:size(1)-l-1)
               t = t:narrow(1, l+2, t:size(1)-l-1)
            end
         end
      end
   end
end

function rnntest.VariableLength_ToSamples()
   local nSamples = 10
   local maxLength = 20
   for run=1,10 do
      local lengths = torch.LongTensor(nSamples)
      lengths:random(maxLength)
      local samples = {}
      for i=1,nSamples do
         samples[i] = torch.rand(lengths[i], 5)
      end
      local output = torch.Tensor()
      local mask = torch.ByteTensor()
      local indexes, mappedLengths = output.nn.VariableLength_FromSamples(samples, output, mask)
      local new_samples = output.nn.VariableLength_ToSamples(indexes, mappedLengths, output)
      mytester:assert(#samples == #new_samples)
      for i=1,nSamples do
         mytester:assertTensorEq(samples[i], new_samples[i])
      end
   end
end

function rnntest.VariableLength_ToFinal()
   local nSamples = 10
   local maxLength = 20
   for run=1,10 do
      local lengths = torch.LongTensor(nSamples)
      lengths:random(maxLength)
      local samples = {}
      for i=1,nSamples do
         local t = torch.rand(lengths[i], 5)
         samples[i] = t
      end
      local output = torch.Tensor()
      local mask = torch.ByteTensor()
      local indexes, mappedLengths = output.nn.VariableLength_FromSamples(samples, output, mask)

      local final = torch.Tensor()
      output.nn.VariableLength_ToFinal(indexes, mappedLengths, output, final)

      for i=1,nSamples do
         mytester:assertTensorEq(samples[i]:select(1, lengths[i]), final:select(1, i))
      end
   end
end

function rnntest.VariableLength_FromFinal()
   torch.manualSeed(2)
   local nSamples = 10
   local maxLength = 20
   for run=1,1 do
      local lengths = torch.LongTensor(nSamples)
      lengths:random(maxLength)
      local samples = {}
      for i=1,nSamples do
         local t = torch.rand(lengths[i], 5)
         samples[i] = t
      end
      local output = torch.Tensor()
      local mask = torch.ByteTensor()
      local indexes, mappedLengths = output.nn.VariableLength_FromSamples(samples, output, mask)

      local final = torch.Tensor()
      output.nn.VariableLength_ToFinal(indexes, mappedLengths, output, final)

      local re_output = torch.Tensor()
      output.nn.VariableLength_FromFinal(indexes, mappedLengths, final, re_output)

      local new_samples = output.nn.VariableLength_ToSamples(indexes, mappedLengths, re_output)

      for i=1,nSamples do
         if lengths[i] > 1 then
            mytester:assert(new_samples[i]:narrow(1, 1, lengths[i]-1):abs():sum() < 0.000001)
         end
         mytester:assertTensorEq(samples[i]:select(1, lengths[i]), new_samples[i]:select(1, lengths[i]))
      end
   end
end

function rnntest.VariableLength_lstm()
   -- test seqlen x batchsize x hiddensize
   local maxLength = 8
   local batchSize = 3
   local hiddenSize = 5
   local nIndex = 20

   local function testVL(testLM, lastOnly)
      -- VL(LSTM): test forward

      local input = {}
      local lstm, vl, input2, output
      if not testLM then
         for i=1,batchSize do
            input[i] = torch.randn(torch.random(1,maxLength), hiddenSize)
         end

         lstm = nn.SeqLSTM(hiddenSize, hiddenSize):maskZero()

         input2 = torch.Tensor(maxLength, batchSize, hiddenSize):zero()
      else
         for i=1,batchSize do
            input[i] = torch.Tensor(torch.random(1,maxLength)):random(1,nIndex)
         end

         lstm = nn.Sequential()
            :add(nn.LookupTableMaskZero(nIndex, hiddenSize))
            :add(nn.SeqLSTM(hiddenSize, hiddenSize):maskZero())

         input2 = torch.Tensor(maxLength, batchSize):zero()
      end

      vl = nn.VariableLength(lstm:clone(), lastOnly)

      local output = vl:forward(input)

      for i=1,batchSize do
         local seqlen = input[i]:size(1)
         input2:select(2,i):narrow(1,maxLength-seqlen+1,seqlen):copy(input[i])
      end

      lstm:setZeroMask(nn.utils.getZeroMaskSequence(input2))
      local output2 = lstm:forward(input2)

      if not lastOnly then
         for i=1,batchSize do
            local out1 = output[i]
            local seqlen = input[i]:size(1)
            mytester:assert(out1:size(1) == seqlen)
            local out2 = output2:select(2,i):narrow(1,maxLength-seqlen+1,seqlen)
            mytester:assertTensorEq(out1, out2, 0.00000001)
         end
      else
         mytester:assertTensorEq(output, output2[maxLength], 0.000001)
      end

      -- VL(LSTM): test backward

      local gradOutput, gradOutput2
      if not lastOnly then
         gradOutput = {}
         for i=1,batchSize do
            gradOutput[i] = torch.randn(output[i]:size())
         end

         gradOutput2 = torch.Tensor(maxLength, batchSize, hiddenSize):zero()
         for i=1,batchSize do
            local seqlen = gradOutput[i]:size(1)
            gradOutput2:select(2,i):narrow(1,maxLength-seqlen+1,seqlen):copy(gradOutput[i])
         end
      else
         gradOutput = torch.randn(batchSize, hiddenSize)
         gradOutput2 = torch.Tensor(maxLength, batchSize, hiddenSize):zero()
         gradOutput2[maxLength]:copy(gradOutput)
      end

      vl:zeroGradParameters()
      local gradInput = vl:backward(input, gradOutput)

      for i=1,batchSize do
         mytester:assert(input[i]:isSameSizeAs(gradInput[i]))
      end

      lstm:zeroGradParameters()
      local gradInput2 = lstm:backward(input2, gradOutput2)

      for i=1,batchSize do
         local gradIn1 = gradInput[i]
         local seqlen = input[i]:size(1)
         mytester:assert(gradIn1:size(1) == seqlen)
         local gradIn2 = gradInput2:select(2,i):narrow(1,maxLength-seqlen+1,seqlen)
         mytester:assertTensorEq(gradIn1, gradIn2, 0.00000001)
      end

      local params, gradParams = vl:parameters()
      local params2, gradParams2 = lstm:parameters()

      for i=1,#params2 do
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001)
      end
   end

   -- testVL(testLstm, lastOnly)
   testVL(false, false)
   testVL(true, false)
   testVL(false, true)
   testVL(true, true)
end

function rnntest.StepLSTM()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5
   local steplstm = nn.StepLSTM(inputsize, outputsize)
   local stepmodule = nn.Sequential()
      :add(nn.FlattenTable())
      :add(steplstm)
   local recmodule = nn.Sequential()
      :add(nn.Recurrence(stepmodule, {{outputsize}, {outputsize}}, 1, seqlen))
      :add(nn.SelectTable(1))
   local lstm = nn.Sequencer(recmodule)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = lstm:forward(input)

   local seqlstm = nn.SeqLSTM(inputsize, outputsize)
   seqlstm.weight:copy(steplstm.weight)
   seqlstm.bias:copy(steplstm.bias)

   local output2 = seqlstm:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   lstm:zeroGradParameters()
   seqlstm:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = lstm:backward(input, gradOutput)

   local gradInput2 = seqlstm:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = seqlstm:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.StepLSTM_C()
   local batchsize, inputsize, outputsize = 2, 3, 4
   local steplstm = nn.StepLSTM(inputsize, outputsize)
   local steplstm2 = steplstm:clone()
   steplstm2.forceLua = true

   local input = {torch.randn(batchsize, inputsize), torch.randn(batchsize, outputsize), torch.randn(batchsize, outputsize)}
   local output = steplstm:forward(input)
   local output2 = steplstm2:forward(input)

   mytester:assertTensorEq(output[1], output2[1], 0.000001)
   mytester:assertTensorEq(output[2], output2[2], 0.000001)
   mytester:assertTensorEq(steplstm.gates, steplstm2.gates, 0.000001)

   steplstm:zeroGradParameters()
   steplstm2:zeroGradParameters()

   local gradOutput = {torch.randn(batchsize, outputsize), torch.randn(batchsize, outputsize)}
   local gradInput = steplstm:backward(input, gradOutput, 1)
   local gradGates = torch.getBuffer('StepLSTM', 'grad_gates', steplstm.gates):clone()
   local gradInput2 = steplstm2:backward(input, gradOutput, 1)
   local gradGates2 = torch.getBuffer('StepLSTM', 'grad_gates', steplstm.gates)

   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001)
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001)
   mytester:assertTensorEq(gradInput[3], gradInput2[3], 0.000001)
   mytester:assertTensorEq(gradGates, gradGates2, 0.000001)

   local params, gradParams = steplstm:parameters()
   local params2, gradParams2 = steplstm2:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.StepLSTM_projection()
   local seqlen, batchsize = 2, 3
   local inputsize, hiddensize, outputsize = 4, 5, 6
   local steplstm = nn.StepLSTM(inputsize, hiddensize, outputsize)
   local stepmodule = nn.Sequential()
      :add(nn.FlattenTable())
      :add(steplstm)
   local recmodule = nn.Sequential()
      :add(nn.Recurrence(stepmodule, {{outputsize}, {hiddensize}}, 1, seqlen))
      :add(nn.SelectTable(1))
   local lstm = nn.Sequencer(recmodule)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = lstm:forward(input)

   local seqlstm = nn.SeqLSTMP(inputsize, hiddensize, outputsize)
   seqlstm.weight:copy(steplstm.weight)
   seqlstm.bias:copy(steplstm.bias)
   seqlstm.weightO:copy(steplstm.weightO)

   local output2 = seqlstm:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   lstm:zeroGradParameters()
   seqlstm:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = lstm:backward(input, gradOutput)

   local gradInput2 = seqlstm:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = seqlstm:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.StepLSTM_C_projection()
   local batchsize, inputsize, hiddensize, outputsize = 2, 3, 4, 5
   local steplstm = nn.StepLSTM(inputsize, hiddensize, outputsize)
   local steplstm2 = steplstm:clone()
   steplstm2.forceLua = true

   local input = {torch.randn(batchsize, inputsize), torch.randn(batchsize, outputsize), torch.randn(batchsize, hiddensize)}
   local output = steplstm:forward(input)
   local output2 = steplstm2:forward(input)

   mytester:assertTensorEq(output[1], output2[1], 0.000001)
   mytester:assertTensorEq(output[2], output2[2], 0.000001)
   mytester:assertTensorEq(steplstm.gates, steplstm2.gates, 0.000001)
   mytester:assertTensorEq(steplstm.hidden, steplstm2.hidden, 0.000001)

   steplstm:zeroGradParameters()
   steplstm2:zeroGradParameters()

   local gradOutput = {torch.randn(batchsize, outputsize), torch.randn(batchsize, hiddensize)}
   local gradInput = steplstm:backward(input, gradOutput, 1)
   local gradGates = torch.getBuffer('StepLSTM', 'grad_gates', steplstm.gates):clone()
   local gradInput2 = steplstm2:backward(input, gradOutput, 1)
   local gradGates2 = torch.getBuffer('StepLSTM', 'grad_gates', steplstm.gates)

   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001)
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001)
   mytester:assertTensorEq(gradInput[3], gradInput2[3], 0.000001)
   mytester:assertTensorEq(gradGates, gradGates2, 0.000001)

   local params, gradParams = steplstm:parameters()
   local params2, gradParams2 = steplstm2:parameters()

   for i=1,3 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.StepGRU()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5
   local stepmodule = nn.StepGRU(inputsize, outputsize)
   local recmodule = nn.Recurrence(stepmodule, outputsize, 1, seqlen)
   local gru = nn.Sequencer(recmodule)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = gru:forward(input)

   local seqgru = nn.SeqGRU(inputsize, outputsize)
   seqgru.weight:copy(stepmodule.weight)
   seqgru.bias:copy(stepmodule.bias)

   local output2 = seqgru:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   gru:zeroGradParameters()
   seqgru:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = gru:backward(input, gradOutput)

   local gradInput2 = seqgru:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = gru:parameters()
   local params2, gradParams2 = seqgru:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.StepGRU_C()
   local batchsize, inputsize, outputsize = 2, 3, 4
   local stepgru = nn.StepGRU(inputsize, outputsize)
   local stepgru2 = stepgru:clone()
   stepgru2.forceLua = true

   local input = {torch.randn(batchsize, inputsize), torch.randn(batchsize, outputsize), torch.randn(batchsize, outputsize)}
   local output = stepgru:forward(input)
   local output2 = stepgru2:forward(input)

   mytester:assertTensorEq(output, output2, 0.000001)
   mytester:assertTensorEq(stepgru.gates, stepgru2.gates, 0.000001)

   stepgru:zeroGradParameters()
   stepgru2:zeroGradParameters()

   local gradOutput = torch.randn(batchsize, outputsize)
   local gradInput = stepgru:backward(input, gradOutput, 1)
   local gradGates = torch.getBuffer('StepGRU', 'grad_gates', stepgru.gates):clone()
   local gradInput2 = stepgru2:backward(input, gradOutput, 1)
   local gradGates2 = torch.getBuffer('StepGRU', 'grad_gates', stepgru.gates)

   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001)
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001)
   mytester:assertTensorEq(gradGates, gradGates2, 0.000001)

   local params, gradParams = stepgru:parameters()
   local params2, gradParams2 = stepgru2:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.RecLSTM()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5
   local reclstm = nn.RecLSTM(inputsize, outputsize)
   local lstm = nn.Sequencer(reclstm)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = lstm:forward(input)

   local seqlstm = nn.SeqLSTM(inputsize, outputsize)
   seqlstm.weight:copy(reclstm.modules[1].weight)
   seqlstm.bias:copy(reclstm.modules[1].bias)

   local output2 = seqlstm:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   lstm:zeroGradParameters()
   seqlstm:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = lstm:backward(input, gradOutput)

   local gradInput2 = seqlstm:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = seqlstm:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.RecLSTM_projection()
   local seqlen, batchsize = 3, 4
   local inputsize, hiddensize, outputsize = 2, 6, 5
   local reclstm = nn.RecLSTM(inputsize, hiddensize, outputsize)
   local lstm = nn.Sequencer(reclstm)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = lstm:forward(input)

   local seqlstm = nn.SeqLSTM(inputsize, hiddensize, outputsize)
   seqlstm.weight:copy(reclstm.modules[1].weight)
   seqlstm.bias:copy(reclstm.modules[1].bias)
   seqlstm.weightO:copy(reclstm.modules[1].weightO)

   local output2 = seqlstm:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   lstm:zeroGradParameters()
   seqlstm:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = lstm:backward(input, gradOutput)

   local gradInput2 = seqlstm:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = seqlstm:parameters()

   for i=1,3 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.RecLSTM_maskzero()
   local T, N, D, H = 3, 2, 4, 5
   local reclstm = nn.RecLSTM(D,H):maskZero()
   local seqlstm = nn.Sequencer(reclstm)
   local seqlstm2 = nn.SeqLSTM(D,H):maskZero()
   seqlstm2.weight:copy(reclstm.modules[1].weight)
   seqlstm2.bias:copy(reclstm.modules[1].bias)

   local input = torch.randn(T, N, D)
   input[{2,1}]:fill(0)
   input[{3,2}]:fill(0)
   local gradOutput = torch.randn(T, N, H)

   local zeroMask = torch.ByteTensor(T, N):zero()
   zeroMask[{2,1}] = 1
   zeroMask[{3,2}] = 1
   seqlstm:setZeroMask(zeroMask)
   seqlstm2:setZeroMask(zeroMask)
   local output = seqlstm:forward(input)
   local output2 = seqlstm2:forward(input)

   mytester:assert(output[{2,1}]:norm() == 0)
   mytester:assert(output[{3,2}]:norm() == 0)

   mytester:assertTensorEq(output, output2, 0.000001)

   seqlstm:zeroGradParameters()
   local gradInput = seqlstm:backward(input, gradOutput)
   seqlstm2:zeroGradParameters()
   local gradInput2 = seqlstm2:backward(input, gradOutput)

   mytester:assert(gradInput[{2,1}]:norm() == 0)
   mytester:assert(gradInput[{3,2}]:norm() == 0)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqlstm:parameters()
   local params2, gradParams2 = seqlstm2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParams[i], 0.0000001)
   end
end

function rnntest.RecGRU()
   local seqlen, batchsize = 3, 4
   local inputsize, outputsize = 2, 5
   local recgru = nn.RecGRU(inputsize, outputsize)
   local gru = nn.Sequencer(recgru)

   local input = torch.randn(seqlen, batchsize, inputsize)
   local output = gru:forward(input)

   local seqgru = nn.SeqGRU(inputsize, outputsize)
   seqgru.weight:copy(recgru.modules[1].weight)
   seqgru.bias:copy(recgru.modules[1].bias)

   local output2 = seqgru:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001)

   gru:zeroGradParameters()
   seqgru:zeroGradParameters()

   local gradOutput = torch.randn(seqlen, batchsize, outputsize)
   local gradInput = gru:backward(input, gradOutput)

   local gradInput2 = seqgru:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = gru:parameters()
   local params2, gradParams2 = seqgru:parameters()

   for i=1,#params2 do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end
end

function rnntest.RecGRU_maskzero()
   local T, N, D, H = 3, 2, 4, 5
   local recgru = nn.RecGRU(D,H):maskZero()
   local seqgru = nn.Sequencer(recgru)
   local seqgru2 = nn.SeqGRU(D,H):maskZero()
   seqgru2.weight:copy(recgru.modules[1].weight)
   seqgru2.bias:copy(recgru.modules[1].bias)

   local input = torch.randn(T, N, D)
   input[{2,1}]:fill(0)
   input[{3,2}]:fill(0)
   local gradOutput = torch.randn(T, N, H)

   local zeroMask = torch.ByteTensor(T, N):zero()
   zeroMask[{2,1}] = 1
   zeroMask[{3,2}] = 1
   seqgru:setZeroMask(zeroMask)
   seqgru2:setZeroMask(zeroMask)
   local output = seqgru:forward(input)
   local output2 = seqgru2:forward(input)

   mytester:assert(output[{2,1}]:norm() == 0)
   mytester:assert(output[{3,2}]:norm() == 0)

   mytester:assertTensorEq(output, output2, 0.000001)

   seqgru:zeroGradParameters()
   local gradInput = seqgru:backward(input, gradOutput)
   seqgru2:zeroGradParameters()
   local gradInput2 = seqgru2:backward(input, gradOutput)

   mytester:assert(gradInput[{2,1}]:norm() == 0)
   mytester:assert(gradInput[{3,2}]:norm() == 0)

   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)

   local params, gradParams = seqgru:parameters()
   local params2, gradParams2 = seqgru2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParams[i], 0.0000001)
   end
end

function rnntest.LinearRNN()
   local inputsize, outputsize = 3, 4
   local seqlen, batchsize = 5, 2

   local input = torch.randn(seqlen, batchsize, inputsize)
   local gradOutput = torch.randn(seqlen, batchsize, outputsize)

   local lrnn = nn.Sequencer(nn.LinearRNN(inputsize, outputsize))

   local output = lrnn:forward(input)
   lrnn:zeroGradParameters()
   local gradInput = lrnn:backward(input, gradOutput)

   mytester:assert(output:isSameSizeAs(gradOutput))
   mytester:assert(gradInput:isSameSizeAs(input))

   local params, gradParams = lrnn:parameters()
   for i=1,2 do
      mytester:assert(gradParams[i]:abs():mean() > 0.000001)
   end
end

function rnntest.LookupRNN()
   local nindex, outputsize = 3, 4
   local seqlen, batchsize = 5, 2

   local input = torch.LongTensor(seqlen, batchsize):random(1,nindex)
   local gradOutput = torch.randn(seqlen, batchsize, outputsize)

   local lrnn = nn.Sequencer(nn.LookupRNN(nindex, outputsize))

   local output = lrnn:forward(input)
   lrnn:zeroGradParameters()
   lrnn:backward(input, gradOutput)

   mytester:assert(output:isSameSizeAs(gradOutput))

   local params, gradParams = lrnn:parameters()
   for i=1,2 do
      mytester:assert(gradParams[i]:abs():mean() > 0.000001)
   end
end


function rnntest.Module_sharedClone()

   local function testrnn(mlp, name)
      mlp:zeroGradParameters()
      local mlp = mlp:clone()
      local clone = mlp:clone():sharedClone(true, true)

      for i=1,2 do
         local input = torch.randn(2,3)
         local gradOutput = torch.randn(2,4)

         local output = mlp:forward(input)
         local gradInput = mlp:backward(input, gradOutput)
         local output4 = clone:forward(input)
         local gradInput4 = clone:backward(input, gradOutput)

         mytester:assertTensorEq(output, output4, 0.00001, name.." updateOutput")
         mytester:assertTensorEq(gradInput, gradInput4, 0.00001, name.." updateGradInput")

         mlp:updateParameters(0.1)
         clone:updateParameters(0.1)

         local params, gradParams = mlp:parameters()
         local params2, gradParams2 = clone:parameters()

         mytester:assert(#params == #params2, name.." num params err")
         mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")

         for i,param in ipairs(params) do
            mytester:assertTensorEq(param, params2[i], 0.00001, name.." params2 err "..i)
            mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams2 err "..i)
         end
      end
   end

   local function test(mlp, name)
      mlp:zeroGradParameters()
      local clone = mlp:clone()
      clone:share(mlp,"weight","bias","gradWeight","gradBias") -- this actually won't work for nn.Recurrent

      local mlp2 = mlp:clone() -- not shared with mlp
      local clone2 = mlp2:sharedClone(true, true)
      mlp2.__test = 1
      clone2.__test = 2
      mytester:assert(mlp2.__test ~= clone2.__test)

      local params, gradParams = mlp:parameters()
      local params4, gradParams4 = clone:parameters()
      local params2, gradParams2 = clone2:parameters()
      local params3, gradParams3 = mlp2:parameters()

      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#params3 == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
      mytester:assert(#gradParams == #gradParams3, name.." num gradParams err")

      local input = torch.randn(2,3)
      local gradOutput = torch.randn(2,4)

      local output = mlp:forward(input)
      local gradInput = mlp:backward(input, gradOutput)

      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params4[i], 0.00001, name.." params4  err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams4[i], 0.00001, name.." gradParams4 err "..i)
      end

      local output4 = clone:forward(input)
      local gradInput4 = clone:backward(input, gradOutput)

      mytester:assertTensorEq(output, output4, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput4, 0.00001, name.." updateGradInput")

      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params4[i], 0.00001, name.." params4  err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams4[i], 0.00001, name.." gradParams4 err "..i)
      end

      local output2 = clone2:forward(input)
      local gradInput2 = clone2:backward(input, gradOutput)

      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")

      for i,param in ipairs(params) do
         mytester:assertTensorEq(params2[i], params3[i], 0.00001, name.." params 2 3  err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.00001, name.." gradParams 2 3 err "..i)
      end

      local output3 = mlp2:forward(input)
      local gradInput3 = mlp2:backward(input, gradOutput)

      mytester:assertTensorEq(output3, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput3, gradInput2, 0.00001, name.." updateGradInput")

      for i,param in ipairs(params) do
         mytester:assertTensorEq(params2[i], params3[i], 0.00001, name.." params 2 3  err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.00001, name.." gradParams 2 3 err "..i)
      end

      mlp:updateParameters(0.1)
      mlp2:updateParameters(0.1)

      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params3[i], 0.00001, name.." params3 (mlp vs mlp:clone()) err "..i) -- fail
         mytester:assertTensorEq(gradParams[i], gradParams3[i], 0.00001, name.." gradParams3 err "..i) -- fail
      end
   end

   test(nn.Linear(3,4), 'linear')

   local mlp = nn.Sequential()
   mlp:add(nn.Linear(3,7))
   mlp:add(nn.Tanh())
   mlp:add(nn.Euclidean(7,4))
   mlp:add(nn.LogSoftMax())
   test(mlp, 'sequential')


   local function test2(rnn, name)
      rnn:zeroGradParameters()
      local clone = rnn:sharedClone()

      local input = torch.randn(2,3)
      local gradOutput = torch.randn(2,4)

      local output = rnn:forward(input)
      local gradInput = rnn:backward(input, gradOutput)
      local output2 = clone:forward(input)
      local gradInput2 = clone:backward(input, gradOutput)

      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")

      rnn:updateParameters(0.1)
      clone:updateParameters(0.1)

      local params, gradParams = rnn:parameters()
      local params2, gradParams2 = clone:parameters()

      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")

      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params2[i], 0.00001, name.." params (rnn vs rnn:sharedClone()) err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams (rnn vs rnn:sharedClone()) err "..i)
      end

      local output = rnn:forward(input)
      local gradInput = rnn:backward(input, gradOutput)
      local output2 = clone:forward(input)
      local gradInput2 = clone:backward(input, gradOutput)

      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")

      rnn:updateParameters(0.1)
      clone:updateParameters(0.1)

      local params, gradParams = rnn:parameters()
      local params2, gradParams2 = clone:parameters()

      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")

      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params2[i], 0.00001, name.." params (rnn vs rnn:sharedClone()) err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams (rnn vs rnn:sharedClone()) err "..i)
      end
   end

   local rnn = nn.LinearRNN(3,4)
   testrnn(rnn, 'rnn1')
   local seq = nn.Sequential()
   seq:add(nn.Repeater(nn.LinearRNN(3, 2), 3))
   seq:add(nn.Sequencer(nn.Linear(2,4)))
   seq:add(nn.SelectTable(-1))
   test2(seq, 'rnn2')
   test2(seq, 'rnn3')

   if pcall(function() require 'nngraph' end) then
      local lin1 = nn.Linear(10, 10)
      local p1, gp1 = lin1:getParameters()

      local lin2_ = lin1:clone()

      local x = nn.Identity()()
      local y = lin2_(x)

      local lin2 = nn.gModule({x}, {y})

      local lin3 = lin2:sharedClone()

      local input = torch.randn(4, 10)
      local gradOutput = torch.randn(4, 10)

      lin1:zeroGradParameters()
      lin2:zeroGradParameters()

      local params1, gradParams1 = lin1:parameters()
      local params2, gradParams2 = lin2:parameters()
      local params3, gradParams3 = lin3:parameters()

      local output1 = lin1:forward(input)
      local gradInput1 = lin1:backward(input, gradOutput)
      lin1:updateParameters(0.1)

      local output2 = lin2:forward(input)
      local gradInput2 = lin2:backward(input, gradOutput)
      lin2:updateParameters(0.1)

      mytester:assertTensorEq(output1, output2, 0.000001)
      mytester:assertTensorEq(gradInput1, gradInput2, 0.000001)

      for i=1,#params2 do
         mytester:assertTensorEq(params2[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
         mytester:assertTensorEq(params1[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end

      -- ok now lets forward/backward/update lin1 and lin3 to test sharedClone

      local output1 = lin1:forward(input)
      local gradInput1 = lin1:backward(input, gradOutput)

      local output3 = lin3:forward(input)
      local gradInput3 = lin3:backward(input, gradOutput)

      for i=1,#params2 do
         mytester:assertTensorEq(params2[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
         mytester:assertTensorEq(params1[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end

      mytester:assertTensorEq(output1, output3, 0.000001)
      mytester:assertTensorEq(gradInput1, gradInput3, 0.000001)

      for i=1,#params2 do
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end

   end
end

function rnntest.Module_gradParamClip()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(10,10))
   mlp:add(nn.Euclidean(15,12))
   mlp:add(nn.SpatialConvolution(5,5,5,5))
   mlp:add(nn.LookupTable(100,100))
   local param, gradParam = mlp:getParameters()
   gradParam:uniform(-1,1)
   local norm = gradParam:norm()
   local mlp2 = mlp:clone()
   local cutoff = norm/2
   local norm2 = mlp2:gradParamClip(cutoff)
   mytester:assert(math.abs(norm2-norm) < 0.000001, "Module:gradParamClip norm err "..norm2.." ~= "..norm)
   local shrink_factor = cutoff / norm
   gradParam:mul(shrink_factor)
   local param2, gradParam2 = mlp2:getParameters()
   mytester:assertTensorEq(gradParam, gradParam2, 0.000001, "Module:gradParamClip clip err")

   local norm = gradParam:norm()
   local cutoff = norm*2
   local norm2 = mlp2:gradParamClip(cutoff)
   mytester:assert(math.abs(norm2-norm) < 0.000001, "Module:gradParamClip norm 2 err "..norm2.." ~= "..norm)
   mytester:assertTensorEq(gradParam, gradParam2, 0.000001, "Module:gradParamClip clip 2 err")
end

function rnntest.Module_getParameters()
   -- test that getParameters will preserve parameters sharing for hidden modules
   local lin = nn.Linear(3,4)
   local lin2 = lin:sharedClone()
   lin.sharedClone = lin2
   local params, gradParams = lin:getParameters()
   params:add(-1)
   gradParams:fill(-1)

   local params1, gradParams1 = lin:parameters()
   local params2, gradParams2 = lin2:parameters()

   for i=1,#params1 do
      mytester:assertTensorEq(params1[i], params2[i], 0.000001, "getParameters param err "..i)
      mytester:assertTensorEq(gradParams1[i], gradParams2[i], 0.000001, "getParameters gradParam err "..i)
   end
end

function rnntest.Serial()
   function test(mlp, name)
      local input = torch.randn(4,3)
      local gradOutput = torch.randn(4,7)
      local mlp2 = mlp:clone():Serial()

      local output = mlp:forward(input):clone()
      local gradInput = mlp:backward(input, gradOutput):clone()

      local output2 = mlp2:forward(input)
      local gradInput2 = mlp2:backward(input, gradOutput)

      mytester:assertTensorEq(output, output2, 0.000001, name.." serial forward error")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." serial backward error")

      mlp2:mediumSerial()
      mlp2.tensortype = 'torch.FloatTensor'
      local mlp3 = mlp2:clone()

      mytester:assert(mlp3.modules[1].output:nElement() == 0, name.." serial medium empty err")
      mytester:assert(torch.type(mlp3.modules[1].output) == 'torch.FloatTensor', name.." serial medium type err")

      mlp:zeroGradParameters()
      local output = mlp:forward(input)
      local gradInput = mlp:backward(input, gradOutput)

      mlp3:zeroGradParameters()
      local output2 = mlp3:forward(input:float())
      local gradInput2 = mlp3:backward(input:float(), gradOutput:float())

      mytester:assertTensorEq(output:float(), output2, 0.000001, name.." serial forward error")
      mytester:assertTensorEq(gradInput:float(), gradInput2, 0.00001, name.." serial backward error")

      local params, gradParams = mlp:parameters()
      local params2, gradParams2 = mlp3:parameters()
      mytester:assert(#params == #params2)
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param:float(), params2[i], 0.00001, name.." params err "..i)
         mytester:assertTensorEq(gradParams[i]:float(), gradParams2[i], 0.00001, name.." gradParams err "..i)
      end
   end

   local mlp = nn.Sequential():extend(
      nn.Linear(3,4),
      nn.Tanh(),
      nn.Linear(4,5),
      nn.Sequential():extend(
         nn.Linear(5,6),
         nn.Tanh(),
         nn.Linear(6,7)
      )
   )

   test(mlp, 'mlp')

   local seq = nn.Sequential()
   seq:add(nn.Repeater(nn.LinearRNN(3,2), 3))
   seq:add(nn.Sequencer(nn.Linear(2,7)))
   seq:add(nn.SelectTable(-1))
   test(seq, 'rnn2')
end

function rnntest.Convert()
   -- batch mode
   local c = nn.Convert('bchw', 'chwb')
   local input = torch.randn(8,3,5,5)
   local output = c:forward(input)
   local output2 = input:transpose(1,4):transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd bchw->chwb")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Convert bwd bchw->chwb")
   local c = nn.Convert('bchw', 'bf')
   local output = c:forward(input)
   local output2 = input:view(8,-1)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd bchw->bf")
   c:float()
   local output = c:forward(input:float())
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type()")
   local output = c:forward(input)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() double->float")
   -- non-batch mode
   local c = nn.Convert('chw', 'hwc')
   local input = torch.randn(3,5,5)
   local output = c:forward(input)
   local output2 = input:transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd chw->hwc non-batch")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Convert bwd chw->hwc non-batch")
   local c = nn.Convert('chw', 'f')
   local output = c:forward(input)
   local output2 = input:view(-1)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd chw->bf non-batch")
   c:float()
   local output = c:forward(input:float())
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() non-batch")
   local output = c:forward(input)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() double->float non-batch")
end

function rnntest.Collapse()
   local c = nn.Collapse(3)
   local input = torch.randn(8,3,4,5)
   local output = c:forward(input)
   mytester:assertTensorEq(input:view(8,-1), output, 0.000001, "Collapse:forward")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Collapse:backward")
   mytester:assertTableEq(gradInput:size():totable(), input:size():totable(), 0.000001, "Collapse:backward size")
   local input2 = input:transpose(1,4)
   local output2 = c:forward(input2)
   mytester:assertTensorEq(input2:contiguous():view(5,-1), output2, 0.000001, "Collapse:forward non-contiguous")
   local gradInput2 = c:backward(input2, output2)
   mytester:assertTensorEq(gradInput2, input2, 0.000001, "Collapse:backward non-contiguous")
   mytester:assertTableEq(gradInput2:size():totable(), input2:size():totable(), 0.000001, "Collapse:backward size non-contiguous")
end

function rnntest.ZipTable()
   -- input : { {a1,a2}, {b1,b2}, {c1,c2} }
   -- output : { {a1,b1,c1}, {a2,b2,c2} }
   local z = nn.ZipTable()
   local input = {
      {torch.randn(3,4), torch.randn(3,4)},
      {torch.randn(3,4), torch.randn(3,4)},
      {torch.randn(3,4), torch.randn(3,4)}
   }
   local output = z:forward(input)
   mytester:assert(#output == 2, "ZipTable #output")
   mytester:assert(#(output[1]) == 3, "ZipTable #output[1]")
   mytester:assertTensorEq(input[1][1], output[1][1], 0.000001, "ZipTable input11")
   mytester:assertTensorEq(input[1][2], output[2][1], 0.000001, "ZipTable input12")
   mytester:assertTensorEq(input[3][2], output[2][3], 0.000001, "ZipTable input32")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 3, "ZipTable #gradInput")
   mytester:assert(#(gradInput[1]) == 2, "ZipTable #gradInput[1]")
   mytester:assertTensorEq(input[1][1], gradInput[1][1], 0.000001, "ZipTable gradInput11")
   mytester:assertTensorEq(input[1][2], gradInput[1][2], 0.000001, "ZipTable gradInput12")
   mytester:assertTensorEq(input[3][2], gradInput[3][2], 0.000001, "ZipTable gradInput32")
end

function rnntest.ZipTableOneToMany()
   -- input : { v, {a,b,c} }
   -- output : { {v,a}, {v,b}, {v,c} }
   local z = nn.ZipTableOneToMany()
   local input = { torch.randn(3), { torch.randn(4), torch.rand(4), torch.rand(4) } }
   local output = z:forward(input)
   mytester:assert(#output == 3, "ZipTableOneToMany #output")
   mytester:assert(#(output[1]) == 2, "ZipTableOneToMany #output[1]")
   mytester:assert(#(output[2]) == 2, "ZipTableOneToMany #output[2]")
   mytester:assert(#(output[3]) == 2, "ZipTableOneToMany #output[3]")
   mytester:assertTensorEq(input[1], output[1][1], 0.000001, "ZipTableOneToMany input1 output11")
   mytester:assertTensorEq(input[1], output[2][1], 0.000001, "ZipTableOneToMany input1 output21")
   mytester:assertTensorEq(input[1], output[3][1], 0.000001, "ZipTableOneToMany input1 output31")
   mytester:assertTensorEq(input[2][1], output[1][2], 0.000001, "ZipTableOneToMany input21")
   mytester:assertTensorEq(input[2][2], output[2][2], 0.000001, "ZipTableOneToMany input22")
   mytester:assertTensorEq(input[2][3], output[3][2], 0.000001, "ZipTableOneToMany input23")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 2, "ZipTableOneToMany #gradInput")
   mytester:assert(#(gradInput[2]) == 3, "ZipTableOneToMany #gradInput[2]")
   mytester:assertTensorEq(input[2][1], gradInput[2][1], 0.000001, "ZipTableOneToMany gradInput21")
   mytester:assertTensorEq(input[2][2], gradInput[2][2], 0.000001, "ZipTableOneToMany gradInput22")
   mytester:assertTensorEq(input[2][3], gradInput[2][3], 0.000001, "ZipTableOneToMany gradInput32")
   mytester:assertTensorEq(torch.mul(input[1], 3), gradInput[1], 0.000001, "ZipTableOneToMany gradInput21")
end

function rnntest.CAddTensorTable()
   -- input : { v, {a,b,c} }
   -- output : { v+a, v+b, v+c }
   local z = nn.CAddTensorTable()
   local input = { torch.randn(3), { torch.randn(3), torch.rand(3), torch.rand(3) } }
   local output = z:forward(input)
   mytester:assert(#output == 3, "CAddTensorTable #output")
   mytester:assertTensorEq(input[1]+input[2][1], output[1], 0.00001, "CAddTensorTable input21 output1")
   mytester:assertTensorEq(input[1]+input[2][2], output[2], 0.00001, "CAddTensorTable input22 output2")
   mytester:assertTensorEq(input[1]+input[2][3], output[3], 0.00001, "CAddTensorTable input23 output3")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 2, "CAddTensorTable #gradInput")
   mytester:assert(#(gradInput[2]) == 3, "CAddTensorTable #gradInput[2]")
   mytester:assertTensorEq(output[1], gradInput[2][1], 0.000001, "CAddTensorTable gradInput21")
   mytester:assertTensorEq(output[2], gradInput[2][2], 0.000001, "CAddTensorTable gradInput22")
   mytester:assertTensorEq(output[3], gradInput[2][3], 0.000001, "CAddTensorTable gradInput23")
   mytester:assertTensorEq(output[1]+output[2]+output[3], gradInput[1], 0.000001, "CAddTensorTable gradInput1")
end

function rnntest.ReverseTable()
   -- input : { a, b, c, d }
   -- output : { c, b, a, d }
   local r = nn.ReverseTable()
   local input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
   local output = r:forward(input)

   mytester:assert(#output == 4, "ReverseTable #output")
   local k = 1
   for i=#input,1,-1 do
      mytester:assertTensorEq(input[i], output[k], 0.00001, "ReverseTable output err "..k)
      k = k + 1
   end

   local gradInput = r:backward(input, output)
   mytester:assert(#gradInput == 4, "ReverseTable #gradInput")
   for i=1,#input do
      mytester:assertTensorEq(gradInput[i], input[i], 0.00001, "ReverseTable gradInput err "..i)
   end
end

function rnntest.Inception()
   local size = {8,3,32,32}
   local outputSize = {8,16+24+8+12,32,32}
   local input = torch.rand(unpack(size))
   local gradOutput = torch.randn(unpack(outputSize))
   local incep = nn.Inception{inputSize=3, outputSize={16,24}, reduceSize={14,16,8,12}}
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception init error')
   end
   local output = incep:forward(input)
   mytester:assertTableEq(output:size():totable(), outputSize, 0.00001)
   mytester:assert(_.isFinite(output:sum()))
   incep:zeroGradParameters()
   local gradInput = incep:backward(input, gradOutput)
   mytester:assertTableEq(gradInput:size():totable(), size, 0.00001)
   mytester:assert(_.isFinite(gradInput:sum()))
   incep:updateParameters(0.1)
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception update error')
   end
   incep:maxParamNorm(1)
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception maxNorm error')
   end
end

function rnntest.SpatialUniformCrop()
   if not pcall(function() require "nnx" end) then return end -- needs the nnx package
   local input = torch.Tensor(8,3,10,10):copy(torch.range(1,8):view(8,1,1,1):expand(8,3,10,10))
   local gradOutput = torch.Tensor(8,3,4,4):copy(torch.range(1,8):view(8,1,1,1):expand(8,3,4,4))
   local sc = nn.SpatialUniformCrop(4)
   local output, gradInput
   for i=1,100 do
      output = sc:forward(input)
      gradInput = sc:backward(input, gradOutput)
   end
   for i=1,8 do
      mytester:assert(math.abs(output[i]:mean() - i) < 0.0001, "SpatialUniformCrop output err "..i)
      mytester:assert(math.abs(gradInput[i]:mean() - ((i*4*4)/(10*10))) < 0.0001, "SpatialUniformCrop gradInput err"..i)
   end

   local input = torch.zeros(1, 1, 120, 120)
   local temp = input[1]:narrow(2, 30, 60):narrow(3, 30, 60)
   temp:fill(1)
   local scale = {}
   scale['min'] = 0.8
   scale['max'] = 1.2

   local layer = nn.SpatialUniformCrop(100, 100, scale)
   local o = layer:forward(input)
   gradInput = layer:backward(input, o)
   mytester:assert(gradInput:max() ~= nil, "SpatialUniformCrop scaling error.")
end

function rnntest.ModuleCriterion()
   local input = torch.randn(8,4)
   local target = torch.randn(8,4)
   local inputModule = nn.Tanh()
   local criterion = nn.MSECriterion()
   local mc = nn.ModuleCriterion(criterion, inputModule)

   local err = mc:forward(input, target)
   local gradInput = mc:backward(input, target)

   local output = inputModule:forward(input)
   local err2 = criterion:forward(output, target)
   local gradOutput = criterion:backward(output, target)
   local gradInput2 = inputModule:backward(input, gradOutput)

   mytester:assert(err == err2, "ModuleCriterion backward err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "ModuleCriterion backward err")
end

function rnntest.ReinforceNormal()
   local input = torch.randn(500,1000) -- means
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(500)
   -- test scalar stdev
   local stdev = 1
   local rn = nn.ReinforceNormal(stdev)
   local output = rn:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceNormal forward size err")
   local outstd = math.sqrt((input - output):pow(2):mean())
   local err = math.abs(outstd - stdev)
   mytester:assert(err < 0.1, "ReinforceNormal forward std err")
   rn:reinforce(reward)
   local gradInput = rn:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   gradInput2:add(-1, input):div(stdev^2)
   local reward2 = reward:view(500,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceNormal backward err")
   -- test input {mean, stdev}
   local mean, stdev = torch.randn(4,10), torch.rand(4,10)
   local input = {mean, stdev}
   local rn = nn.ReinforceNormal()
   local output = rn:updateOutput(input)
   local reward = torch.randn(4)
   rn:reinforce(reward)
   local gradInput = rn:backward(input, gradOutput)
   mytester:assert(mean:isSameSizeAs(output), "ReinforceNormal forward table input - output size err")
   mytester:assert(gradInput[1]:isSameSizeAs(mean), "ReinforceNormal backward table input - mean size err")
   mytester:assert(gradInput[2]:isSameSizeAs(stdev), "ReinforceNormal backward table input - stdev size err")
   local gradStdev = output:clone():add(-1, mean):pow(2)
   local stdev2 = torch.cmul(stdev,stdev)
   gradStdev:add(-1,stdev2)
   stdev2:cmul(stdev):add(0.00000001)
   gradStdev:cdiv(stdev2)
   local reward2 = reward:view(4,1):expandAs(gradStdev)
   gradStdev:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput[2], gradStdev, 0.000001, "ReinforceNormal backward table input - gradStdev err")
end

function rnntest.ReinforceGamma()
   if not pcall(function() require 'randomkit'; require 'cephes' end) then
      return
   end
   local input = torch.rand(500,1000):fill(250) -- shapes
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(500)
   -- test scalar scale
   local scale = 2
   local rn = nn.ReinforceGamma(scale)
   local output = rn:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceGamma forward size err")
   local outmean = torch.mean(output)
   -- expected value of distribution is shape*scale
   local err = math.abs(outmean - torch.mean(torch.mul(input,scale)))
   mytester:assert(err < 0.1, "ReinforceGamma forward mean err")
   rn:reinforce(reward)
   local gradInput = rn:updateGradInput(input, gradOutput)
   local gradInput2 = torch.log(output:clone())
   gradInput2:add(-1, cephes.digamma(input))
   gradInput2:add(-1*torch.log(scale) )
   local reward2 = reward:view(500,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceGamma backward err")
   -- test input {mean, stdev}
   local shape, scale = torch.rand(4,10), torch.rand(4,10)
   local input = {shape, scale}
   local rn = nn.ReinforceGamma()
   local output = rn:updateOutput(input)
   local reward = torch.randn(4)
   rn:reinforce(reward)
   local gradInput = rn:backward(input, gradOutput)
   mytester:assert(shape:isSameSizeAs(output), "ReinforceGamma forward table input - output size err")
   mytester:assert(gradInput[1]:isSameSizeAs(shape), "ReinforceGamma backward table input - mean size err")
   mytester:assert(gradInput[2]:isSameSizeAs(scale), "ReinforceGamma backward table input - stdev size err")
   local gradScale = torch.cdiv(output:clone(), torch.pow(scale,2) )
   gradScale:add( -1, torch.cdiv( shape, scale) )
   local reward2 = reward:view(4,1):expandAs(gradScale)
   gradScale:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput[2], gradScale, 0.000001, "ReinforceGamma backward table input - gradStdev err")
end

function rnntest.ReinforceBernoulli()
   local input = torch.Tensor(1000,10)
   local p = torch.rand(1,10) -- probability of sampling a 1
   input:copy(p:expandAs(input))
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(1000)
   local rb = nn.ReinforceBernoulli()
   local output = rb:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceBernoulli forward size err")
   mytester:assert(output:min() == 0, "ReinforceBernoulli forward min val err")
   mytester:assert(output:max() == 1, "ReinforceBernoulli forward max val err")
   local binary = true
   output:apply(function(x) if not (x == 1 or x == 0) then binary = false end end)
   mytester:assert(binary, "ReinforceBernoulli forward binary val err")
   local p2 = output:mean(1)
   local err = (p - p2):abs():mean()
   mytester:assert(err < 0.05, "ReinforceBernoulli forward p err")
   rb:reinforce(reward)
   local gradInput = rb:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   local div = output:clone():fill(1):add(-1, input):cmul(input)
   gradInput2:add(-1, input):cdiv(div)
   local reward2 = reward:view(1000,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceBernoulli backward err")
end

function rnntest.ReinforceCategorical()
   local input = torch.Tensor(1000,10)
   local p = torch.rand(1,10)
   p:div(p:sum())
   input:copy(p:expandAs(input))
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(1000)
   local rc = nn.ReinforceCategorical()
   local output = rc:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceCategorical forward size err")
   mytester:assert(output:min() == 0, "ReinforceCategorical forward min val err")
   mytester:assert(output:max() == 1, "ReinforceCategorical forward max val err")
   mytester:assert(output:sum() == 1000, "ReinforceCategorical forward sum err")
   local binary = true
   output:apply(function(x) if not (x == 1 or x == 0) then binary = false end end)
   mytester:assert(binary, "ReinforceCategorical forward binary val err")
   local p2 = output:mean(1)
   local err = (p - p2):abs():mean()
   mytester:assert(err < 0.05, "ReinforceCategorical forward p err")
   rc:reinforce(reward)
   local gradInput = rc:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   gradInput2:cdiv(input+0.00000001)
   local reward2 = reward:view(1000,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceCategorical backward err")
end

function rnntest.VRClassReward()
   local input = {torch.randn(13,10):float(), torch.randn(13,1):float()}
   local target = torch.IntTensor(13):random(1,10)
   local rf = nn.Reinforce():float()
   local vrc = nn.VRClassReward(rf):float()
   local err = vrc:forward(input, target)
   local gradInput = vrc:backward(input, target)
   local val, idx = input[1]:max(2)
   local reward = torch.eq(idx:select(2,1):int(), target):float()
   local err2 = -reward:mean()
   mytester:assert(err == err2, "VRClassReward forward err")
   local gradInput2 = nn.MSECriterion():float():backward(input[2], reward)
   mytester:assertTensorEq(gradInput[2], gradInput2, 0.000001, "VRClassReward backward baseline err")
   mytester:assert(math.abs(gradInput[1]:sum()) < 0.000001, "VRClassReward backward class err")

   if pcall(function() require 'cunn' end) then
      local gradInput = {gradInput[1], gradInput[2]}
      input[1], input[2] = input[1]:cuda(), input[2]:cuda()
      target = target:cuda()
      rf:cuda()
      vrc:cuda()

      local err2 = vrc:forward(input, target)
      local gradInput2 = vrc:backward(input, target)

      mytester:assert(math.abs(err - err2) < 0.000001, "VRClassReward forward cuda err")
      mytester:assertTensorEq(gradInput[2], gradInput2[2]:float(), 0.000001, "VRClassReward backward baseline cuda err")
      mytester:assertTensorEq(gradInput[1], gradInput2[1]:float(), 0.000001, "VRClassReward backward class cuda err")
   end
end

function rnntest.BinaryClassReward()
   local input = {torch.Tensor(10), torch.randn(10,1)}
   input[1]:uniform(0,1)
   local target = torch.LongTensor(10):random(0,1)
   local rf = nn.Reinforce()
   local bcr = nn.BinaryClassReward(rf)
   local err = bcr:forward(input, target)
   local gradInput = bcr:backward(input, target)
   local idx = input[1].new():gt(input[1], 0.5)
   local reward = torch.eq(idx:long(), target):double()
   local err2 = -reward:mean()
   mytester:assert(err == err2, "BinaryClassReward forward err")
   local gradInput2 = nn.MSECriterion():backward(input[2], reward)
   mytester:assertTensorEq(gradInput[2], gradInput2, 0.000001, "BinaryClassReward backward baseline err")
   mytester:assertTensorEq(gradInput[1], torch.zeros(input[1]:size()), 0.000001, "BinaryClassReward backward class err")

   -- test agains VRClassReward
   local input2 = {torch.Tensor(10,2):zero(), input[2]}
   local target2 = torch.add(target, 1)
   for i=1,10 do
      input2[1][i][input[1][i] > 0.5 and 2 or 1] = 1
   end
   local rf2 = nn.Reinforce()
   local vrc = nn.VRClassReward(rf2)
   local err2 = vrc:forward(input2, target2)
   mytester:assert(math.abs(err - err2) < 0.0000001)
   local gradInput2 = vrc:backward(input2, target2)
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.0000001)
   mytester:assertTensorEq(rf2.reward, rf.reward, 0.0000001)
end

function rnntest.Clip()
   local input = torch.randn(200,300)
   local gradOutput = torch.randn(200,300)
   local minval, maxval = -0.05, 0.1
   local clip = nn.Clip(minval, maxval)
   local output = clip:forward(input)
   local output2 = input:clone()
   local mask = input.new()
   mask:gt(input, maxval)
   output2[mask:type("torch.ByteTensor")] = maxval
   mask:lt(input, minval)
   output2[mask:type("torch.ByteTensor")] = minval
   mytester:assertTensorEq(output, output2, 0.00001, "Clip forward err")
   local gradInput = clip:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradOutput, 0.00001, "Clip backward err")
end

function rnntest.Constant()
   local input = torch.randn(20,3,7)
   local gradOutput = torch.randn(20,30,6)
   local value = torch.randn(30,6)
   local const = nn.Constant(value:clone(), 2)
   local output = const:forward(input)
   local gradInput = const:backward(input, output)
   local output2 = value:view(1,30,6):expand(20,30,6)
   mytester:assertTensorEq(output2, output, 0.000001, "Constant forward err")
   mytester:assertTensorEq(gradInput, input:zero(), 0.000001, "Constant backward err")
end

function rnntest.SpatialGlimpse()
   if not pcall(function() require "image" end) then return end -- needs the image package
   if not pcall(function() require "nnx" end) then return end -- needs the nnx package
   local batchSize = 1
   local inputSize = {2,8,8}
   local glimpseSize = 4
   local input = torch.Tensor(batchSize, unpack(inputSize))
   input:range(1,input:nElement())
   input:resize(batchSize, unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local outputSize = {batchSize, inputSize[1]*3, glimpseSize, glimpseSize}
   mytester:assertTableEq(output:size():totable(), outputSize, 0.000001, "SpatialGlimpse output size err")

   local input2 = torch.Tensor(unpack(inputSize))
   input2:range(1,input2:nElement())
   input2:resize(unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location2 = torch.Tensor(2):fill(0) -- center patch
   local output2 = sg:forward{input2,location2}
   mytester:assertTensorEq(output2, output[1], 0.00001, "SpatialGlimpse online output depth=1 err")

   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,2,glimpseSize):narrow(4,2,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 5 output depth=1 err")

   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local padSize = math.floor((glimpseSize-1)/2)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+padSize*2, inputSize[3]+padSize*2):zero()
   pad:narrow(3, padSize + 1, inputSize[2]):narrow(4, padSize + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize):narrow(4,1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse top-left 4 output depth=1 err")

   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2 + 1, inputSize[2]):narrow(4, (glimpseSize-1)/2 + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize):narrow(4,1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse top-left 5 output depth=1 err")

   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, math.floor((glimpseSize-1)/2 + 1), inputSize[2]):narrow(4, math.floor((glimpseSize-1)/2 + 1), inputSize[3]):copy(input)
   local output2 = pad:narrow(3,inputSize[2]-1,glimpseSize):narrow(4,inputSize[3]-1,glimpseSize)
   --print('bottom-right', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse bottom-right 4 output depth=1 err")

   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2, inputSize[2]):narrow(4, (glimpseSize-1)/2, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,inputSize[2]-1,glimpseSize):narrow(4,inputSize[3]-1,glimpseSize)
   --print('bottom-right', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse bottom-right 5 output depth=1 err")

   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize, 1)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 1, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradInput = sg:backward({input,location}, output)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 1 error")

   -- test with spatial resampling
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialReSampling{oheight=glimpseSize*2,owidth=glimpseSize*2}
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse backward online loc err")

   -- test with spatial avg pool
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2)
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse avgpool backward online loc err")

   -- test avg pool with cuda
   if not pcall(function() require "cunn" end) then return end -- needs the cunn package
   local input = input:cuda()

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2):cuda()
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse avgpool backward online loc err")

   if false then
      -- benchmark GPU vs CPU
      local location = torch.FloatTensor(32,2):uniform(-1,1)
      local input = torch.FloatTensor(32,3,224,224):uniform(0,1)
      local gradOutput = torch.FloatTensor(32,9,32,32):uniform(0,1)
      local sg = nn.SpatialGlimpse(32, 3, 2):float()
      sg:forward{input,location}
      local a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdCPUtime = a:time().real

      sg:cuda()
      location = location:cuda()
      input = input:cuda()
      gradOutput = gradOutput:cuda()
      sg:forward{input,location}
      a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdGPUtime = a:time().real
      print(fwdGPUtime, fwdCPUtime, fwdCPUtime/fwdGPUtime)
      -- 0.13885092735291  2.0344181060791  14.651815042678
   end
end

function rnntest.SpatialGlimpse_backwardcompat()
   if not pcall(function() require "nnx" end) then return end -- needs the nnx package
   -- this is ugly, but I know this verson of the module works.
   -- So we try to match the newer versions to it
   local SG, parent = torch.class("nn.SG", "nn.Module")

   function SG:__init(size, depth, scale)
      self.size = size -- height == width
      self.depth = depth or 3
      self.scale = scale or 2

      assert(torch.type(self.size) == 'number')
      assert(torch.type(self.depth) == 'number')
      assert(torch.type(self.scale) == 'number')
      parent.__init(self)
      self.gradInput = {torch.Tensor(), torch.Tensor()}
      if self.scale == 2 then
         self.module = nn.SpatialAveragePooling(2,2,2,2)
      else
         self.module = nn.SpatialReSampling{oheight=size,owidth=size}
      end
      self.modules = {self.module}
   end

   -- a bandwidth limited sensor which focuses on a location.
   -- locations index the x,y coord of the center of the output glimpse
   function SG:updateOutput(inputTable)
      assert(torch.type(inputTable) == 'table')
      assert(#inputTable >= 2)
      local input, location = unpack(inputTable)
      input, location = self:toBatch(input, 3), self:toBatch(location, 1)
      assert(input:dim() == 4 and location:dim() == 2)

      self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)

      self._crop = self._crop or self.output.new()
      self._pad = self._pad or input.new()

      for sampleIdx=1,self.output:size(1) do
         local outputSample = self.output[sampleIdx]
         local inputSample = input[sampleIdx]
         local xy = location[sampleIdx]
         -- (-1,-1) top left corner, (1,1) bottom right corner of image
         local x, y = xy:select(1,1), xy:select(1,2)
         -- (0,0), (1,1)
         x, y = (x+1)/2, (y+1)/2

         -- for each depth of glimpse : pad, crop, downscale
         local glimpseSize = math.floor(self.size)
         for depth=1,self.depth do
            local dst = outputSample[depth]
            if depth > 1 then
               glimpseSize = math.floor(glimpseSize*self.scale)
            end

            -- add zero padding (glimpse could be partially out of bounds)
            local padSize = math.floor((glimpseSize-1)/2)
            self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
            local center = self._pad:narrow(2,padSize+1,input:size(3)):narrow(3,padSize+1,input:size(4))
            center:copy(inputSample)

            -- crop it
            local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
            local x, y = math.floor(math.min(h,math.max(0,x*h))), math.floor(math.min(w,math.max(0,y*w)))

            if depth == 1 then
               dst:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
            else
               self._crop:resize(input:size(2), glimpseSize, glimpseSize)
               self._crop:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))

               if torch.type(self.module) == 'nn.SpatialAveragePooling' then
                  local poolSize = glimpseSize/self.size
                  assert(poolSize % 2 == 0)
                  self.modules[1].kW = poolSize
                  self.modules[1].kH = poolSize
                  self.modules[1].dW = poolSize
                  self.modules[1].dH = poolSize
               end
               dst:copy(self.modules[1]:updateOutput(self._crop))
            end
         end
      end

      self.output:resize(input:size(1), self.depth*input:size(2), self.size, self.size)
      self.output = self:fromBatch(self.output, 1)
      return self.output
   end

   function SG:updateGradInput(inputTable, gradOutput)
      local input, location = unpack(inputTable)
      local gradInput, gradLocation = unpack(self.gradInput)
      input, location = self:toBatch(input, 3), self:toBatch(location, 1)
      gradOutput = self:toBatch(gradOutput, 3)

      gradInput:resizeAs(input):zero()
      gradLocation:resizeAs(location):zero() -- no backprop through location

      gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size)

      for sampleIdx=1,gradOutput:size(1) do
         local gradOutputSample = gradOutput[sampleIdx]
         local gradInputSample = gradInput[sampleIdx]
         local xy = location[sampleIdx] -- height, width
         -- (-1,-1) top left corner, (1,1) bottom right corner of image
         local x, y = xy:select(1,1), xy:select(1,2)
         -- (0,0), (1,1)
         x, y = (x+1)/2, (y+1)/2

         -- for each depth of glimpse : pad, crop, downscale
         local glimpseSize = self.size
         for depth=1,self.depth do
            local src = gradOutputSample[depth]
            if depth > 1 then
               glimpseSize = glimpseSize*self.scale
            end

            -- add zero padding (glimpse could be partially out of bounds)
            local padSize = math.floor((glimpseSize-1)/2)
            self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()

            local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
            local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))
            local pad = self._pad:narrow(2, x+1, glimpseSize):narrow(3, y+1, glimpseSize)

            -- upscale glimpse for different depths
            if depth == 1 then
               pad:copy(src)
            else
               self._crop:resize(input:size(2), glimpseSize, glimpseSize)

               if torch.type(self.module) == 'nn.SpatialAveragePooling' then
                  local poolSize = glimpseSize/self.size
                  assert(poolSize % 2 == 0)
                  self.modules[1].kW = poolSize
                  self.modules[1].kH = poolSize
                  self.modules[1].dW = poolSize
                  self.modules[1].dH = poolSize
               end

               pad:copy(self.modules[1]:updateGradInput(self._crop, src))
            end

            -- copy into gradInput tensor (excluding padding)
            gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
         end
      end

      self.gradInput[1] = self:fromBatch(gradInput, 1)
      self.gradInput[2] = self:fromBatch(gradLocation, 1)

      return self.gradInput
   end

   local batchSize = 1
   local inputSize = {2,8,8}
   local glimpseSize = 4
   local input = torch.randn(batchSize, unpack(inputSize))
   input:resize(batchSize, unpack(inputSize))

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local sg2 = nn.SG(glimpseSize, 2)

   for i=1,10 do
      local location = torch.Tensor(batchSize, 2):uniform(-0.9,0.9)
      local output = sg:forward{input,location}
      local output2 = sg2:forward{input,location}
      mytester:assertTensorEq(output, output2, 0.0000001, "SpatialGlimpse err")
   end

end

-- test rectangle-shaped glimpse sampling
function rnntest.SpatialGlimpseRect()
   if not pcall(function() require "image" end) then return end -- needs the image package
   if not pcall(function() require "nnx" end) then return end -- needs the nnx package
   local batchSize = 1
   local inputSize = {2,8,8}

   local glimpseSize = {4,2} -- {height, width}
   local input = torch.Tensor(batchSize, unpack(inputSize))
   input:range(1,input:nElement())
   input:resize(batchSize, unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = (input:size(3)-glimpseSize[1])/2 + 1
   local x0 = (input:size(4)-glimpseSize[2])/2 + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local outputSize = {batchSize, inputSize[1]*3, glimpseSize[1], glimpseSize[2]}
   mytester:assertTableEq(output:size():totable(), outputSize, 0.000001, "SpatialGlimpseRect output size err")

   local input2 = torch.Tensor(unpack(inputSize))
   input2:range(1,input2:nElement())
   input2:resize(unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location2 = torch.Tensor(2):fill(0) -- center patch
   local output2 = sg:forward{input2,location2}
   mytester:assertTensorEq(output2, output[1], 0.00001, "SpatialGlimpseRect online output depth=1 err")

   local glimpseSize = {5,3}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 5 output depth=1 err")

   local glimpseSize = {4,3}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local padSize = {math.floor((glimpseSize[1]-1)/2), math.floor((glimpseSize[2]-1)/2)}
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+padSize[1]*2, inputSize[3]+padSize[2]*2):zero()
   pad:narrow(3, padSize[1] + 1, inputSize[2]):narrow(4, padSize[2] + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize[1]):narrow(4,1,glimpseSize[2])
   --print('top-left', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect top-left 4 output depth=1 err")

   local glimpseSize = {5,4}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1]-1)/2) + 1
   local x0 = math.floor((glimpseSize[2]-1)/2) + 1
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize[1]):narrow(4,1,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect top-left 5 output depth=1 err")

   local glimpseSize = {3,4}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1]-1)/2) + 1
   local x0 = math.floor((glimpseSize[2]-1)/2) + 1
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local dy = math.floor((glimpseSize[1])/2)
   local dx = math.floor((glimpseSize[2])/2)
   local output2 = pad:narrow(3,inputSize[2]-dy+1,glimpseSize[1]):narrow(4,inputSize[3]-dx+1,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect bottom-right 4 output depth=1 err")

   local glimpseSize = {4,5}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1])/2)
   local x0 = math.floor((glimpseSize[2])/2)
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local dy = math.floor((glimpseSize[1])/2)
   local dx = math.floor((glimpseSize[2])/2)
   local output2 = pad:narrow(3,inputSize[2]-dy+1,glimpseSize[1]):narrow(4,inputSize[3]-dx+1,glimpseSize[2])
   --print('bottom-right', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect bottom-right 5 output depth=1 err")

   -- test gradients
   local glimpseSize = {4,4} -- {height, width}
   local sg = nn.SpatialGlimpse(glimpseSize, 1)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 1, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradInput = sg:backward({input,location}, output)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 1 error")

   -- test with spatial resampling
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialReSampling{oheight=glimpseSize[2]*2,owidth=glimpseSize[1]*2}
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   --print('SpatialReSampling', gradInput2, gradInput[1])
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   --print('SpatialReSampling', gradInput2, gradInput[1])
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect backward online loc err")

   -- test with spatial avg pool
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2)
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect avgpool backward online loc err")

   -- test avg pool with cuda
   if not pcall(function() require "cunn" end) then return end -- needs the cunn package
   local input = input:cuda()

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2):cuda()
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 full error")

   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect avgpool backward online loc err")

   if false then
      -- benchmark GPU vs CPU
      local location = torch.FloatTensor(32,2):uniform(-1,1)
      local input = torch.FloatTensor(32,3,224,224):uniform(0,1)
      local gradOutput = torch.FloatTensor(32,9,32,32):uniform(0,1)
      local sg = nn.SpatialGlimpse({32,24}, 3, 2):float()
      sg:forward{input,location}
      local a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdCPUtime = a:time().real

      sg:cuda()
      location = location:cuda()
      input = input:cuda()
      gradOutput = gradOutput:cuda()
      sg:forward{input,location}
      a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdGPUtime = a:time().real
      print(fwdGPUtime, fwdCPUtime, fwdCPUtime/fwdGPUtime)
      --
   end
end

function rnntest.ArgMax()
   local inputSize = 5
   local batchSize = 3
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize):long()
   local am = nn.ArgMax(1,1)
   local output = am:forward(input)
   local gradInput = am:backward(input, gradOutput)
   local val, idx = torch.max(input, 2)
   mytester:assertTensorEq(idx:select(2,1), output, 0.000001, "ArgMax output asLong err")
   mytester:assertTensorEq(gradInput, input:clone():zero(), 0.000001, "ArgMax gradInput asLong err")
   local am = nn.ArgMax(1,1,false)
   local output = am:forward(input)
   local gradInput = am:backward(input, gradOutput)
   local val, idx = torch.max(input, 2)
   mytester:assertTensorEq(idx:select(2,1):double(), output, 0.000001, "ArgMax output not asLong err")
   mytester:assertTensorEq(gradInput, input:clone():zero(), 0.000001, "ArgMax gradInput not asLong err")
end

function rnntest.CategoricalEntropy()
   local inputSize = 5
   local batchSize = 10
   local minEntropy = 12
   local input_ = torch.randn(batchSize, inputSize)
   local input = nn.SoftMax():updateOutput(input_)
   local gradOutput = torch.Tensor(batchSize, inputSize):zero()
   local ce = nn.CategoricalEntropy()
   local output = ce:forward(input)
   mytester:assertTensorEq(input, output, 0.0000001, "CategoricalEntropy output err")
   local gradInput = ce:backward(input, gradOutput)
   local output2 = input:sum(1)[1]
   output2:div(output2:sum())
   local log2 = torch.log(output2 + 0.000001)
   local entropy2 = -output2:cmul(log2):sum()
   mytester:assert(math.abs(ce.entropy - entropy2) < 0.000001, "CategoricalEntropy entropy err")
   local gradEntropy2 = log2:add(1) -- -1*(-1 - log(p(x))) = 1 + log(p(x))
   gradEntropy2:div(input:sum())
   local gradInput2 = gradEntropy2:div(batchSize):view(1,inputSize):expandAs(input)
   mytester:assertTensorEq(gradInput2, gradInput, 0.000001, "CategoricalEntropy gradInput err")
end

function rnntest.TotalDropout()
   local batchSize = 4
   local inputSize = 3
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize, inputSize)
   local td = nn.TotalDropout()
   local nOne = 0
   for i=1,10 do
      local output = td:forward(input)
      local gradInput = td:backward(input, gradOutput)
      if td.noise == 0 then
         mytester:assert(output:sum() == 0, "TotalDropout forward 0 err")
         mytester:assert(gradInput:sum() == 0, "TotalDropout backward 0 err")
      else
         mytester:assertTensorEq(output, input, 0.000001, "TotalDropout forward 1 err")
         mytester:assertTensorEq(gradInput, gradOutput, 0.000001, "TotalDropout backward 1 err")
         nOne = nOne + 1
      end
   end
   mytester:assert(nOne < 10 and nOne > 1, "TotalDropout bernoulli error")
end


-- Unit Test WhiteNoise
function rnntest.WhiteNoise()
   local input = torch.zeros(3, 28, 28)
   local addNoise = nn.WhiteNoise()
   local output = addNoise:forward(input)
   local meanValue = output:mean()
   local stdValue = output:std()
   mytester:assert(meanValue > -0.01 and meanValue < 0.01)
   mytester:assert(stdValue < 0.15 and stdValue >= 0)

   -- Evaluate
   addNoise:evaluate()
   output = addNoise:forward(input)
   meanValue = output:mean()
   stdValue = output:std()
   mytester:assert(meanValue == 0)
   mytester:assert(stdValue == 0)

   -- backprop
   addNoise:training()
   local gradOutput = torch.rand(3, 28, 28)
   local gradInput = addNoise:updateGradInput(input, gradOutput)
   mytester:assertTensorEq(gradOutput, gradInput, 0.000001, "WhiteNoise backward err")
end

-- Unit Test SpatialBinaryLogisticRegression criterion
function rnntest.SpatialBinaryLogisticRegression()
   local crit = nn.SpatialBinaryLogisticRegression()
   local k = 32
   local h = 28
   local w = 28

   -- Working with batch of images
   local input = torch.zeros(k, 1, h, w)
   local target = torch.zeros(k, 1, h, w)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))/2
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "SpatialBinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1][1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(2*k*h*w)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "SpatialBinaryLogisticRegression gradInput error.")
      end
   end

   -- Working with single image
   k = 1
   local input = torch.zeros(1, h, w)
   local target = torch.zeros(1, h, w)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))/2
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "SpatialBinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(2*k*h*w)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "SpatialBinaryLogisticRegression gradInput error.")
      end
   end
end

-- Unit Test BinaryLogisticRegression criterion
function rnntest.BinaryLogisticRegression()
   local crit = nn.BinaryLogisticRegression()
   local k = 32

   -- Working with batch of images
   local input = torch.zeros(k, 1)
   local target = torch.zeros(k, 1)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "BinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(k)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "BinaryLogisticRegression gradInput error.")
      end
   end

   -- Working nElements not matching.
   local input = torch.zeros(1, k)
   local target = torch.zeros(k, 1)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "BinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(k)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "BinaryLogisticRegression gradInput error.")
      end
   end
end

-- Unit Test SpatialRegionDropout
function rnntest.SpatialRegionDropout()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local p = 0.2
   local value = 2
   local model = nn.SpatialRegionDropout(p)
   local input = torch.zeros(3, 100, 100):fill(value)

   for _, useCuda in pairs(useCudas) do
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      mytester:assert( output:mean() >= value-precision and
                       output:mean() <= value+precision,
                       "SpatialRegionDropout forward mean value incorrect.")

      local gradInput = model:backward(input, input)
      mytester:assert( gradInput:mean() >= value-precision and
                       gradInput:mean() <= value+precision,
                       "SpatialRegionDropout backward mean value incorrect.")
   end
end

-- Unit Test SpatialBinaryConvolution
function rnntest.SpatialBinaryConvolution()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local nInputPlane = 3
   local nOutputPlane = 16
   local kW = 3
   local kH = 3
   local height = 224
   local width = 224

   local model = nn.SpatialBinaryConvolution(nInputPlane, nOutputPlane,
                                             kW, kH)
   local input = torch.rand(nInputPlane, height, width)

   for _, useCuda in pairs(useCudas) do
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      model:zeroGradParameters()
      local output = model:forward(input)
      local gradInput = model:backward(input, output)
   end
end

-- Unit Test SimpleColorTransform
function rnntest.SimpleColorTransform()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local value = 10
   local rangeValue = 2
   local precision = rangeValue*0.1
   local range = torch.zeros(3):fill(rangeValue)
   local model = nn.SimpleColorTransform(3, range)
   local input = torch.zeros(32, 3, 100, 100):fill(value)

   for _, useCuda in pairs(useCudas) do
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      mytester:assert(output:std() <= rangeValue+precision,
                       "SimpleColorTransform output value incorrect.")
      local gradInput = model:backward(input, input)
      mytester:assert(gradInput:sum() == input:sum(),
                       "SimpleColorTransform gradInput value incorrect.")
   end
end

-- Unit Test PCAColorTransform
function rnntest.PCAColorTransform()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local std = 0.1
   local value = 145
   local rangeValue = 1800
   local precision = rangeValue * 3 * std
   local eigenVectors = torch.Tensor({{ 0.58786434,  0.56388045,  0.58004685},
                                      {-0.65427388, -0.0902746 ,  0.75085031},
                                      {-0.47575331,  0.82090763, -0.31586303}})
   local eigenValues = torch.Tensor({4491.21, 722.85, 68.07})
   local model = nn.PCAColorTransform(3, eigenVectors, eigenValues, std)
   local input = torch.zeros(32, 3, 100, 100):fill(value)

   for _, useCuda in pairs(useCudas) do
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      mytester:assert(output:std() <= rangeValue+precision,
                       "PCAColorTransform output value incorrect.")
      local gradInput = model:backward(input, input)
      mytester:assert(gradInput:sum() == input:sum(),
                       "PCAColorTransform gradInput value incorrect.")
   end
end

-- Unit Test FireModule
function rnntest.FireModule()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local activations = {'ReLU', 'Tanh', 'Sigmoid'}
   local nInputPlane = 3
   local width = 32
   local height = 32
   local s1x1 = 16
   local e1x1 = 16
   local e3x3 = 16
   for _, activation in pairs(activations) do
      for _, useCuda in pairs(useCudas) do
         local model = nn.FireModule(nInputPlane, s1x1, e1x1, e3x3)
         local input = torch.rand(1, nInputPlane, height, width)
         if useCuda then
            model:cuda()
            input = input:cuda()
         end
         local output = model:forward(input)
         local gradInput = model:backward(input, output)
      end
   end
end

-- Unit Test SpatialFeatNormalization
function rnntest.SpatialFeatNormalization()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local input = torch.zeros(3, 32, 32):fill(2)
   local mean = torch.zeros(3):fill(1)
   local std = torch.zeros(3):fill(0.5)
   local outputValue = 2
   local gradValue = 4
   for _, useCuda in pairs(useCudas) do
      local model = nn.SpatialFeatNormalization(mean, std)
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      local gradInput = model:backward(input, output)
      mytester:assert( output:mean() == outputValue,
                     "SpatialFeatNormalization forward mean value incorrect.")
      mytester:assert( gradInput:mean() == gradValue,
                     "SpatialFeatNormalization backward mean value incorrect.")
   end
end

function rnntest.OneHot()
   local nClass = 10

   -- batch mode
   local batchSize = 3
   local input = torch.LongTensor(batchSize):random(1, nClass)
   local gradOutput = torch.randn(batchSize, nClass)

   local oh = nn.OneHot(nClass)

   local output = oh:forward(input)
   local output2 = torch.Tensor(batchSize, nClass):zero()
   local eye = torch.eye(nClass)
   output2:index(eye, 1, input)
   mytester:assertTensorEq(output, output2, 0.000001, "OneHot forward batch err")
   mytester:assert(output:dim() == 2)

   -- non-batch mode (number input)
   local num = 3
   local output3 = torch.zeros(nClass)
   output3[num] = 1.0
   mytester:assertTensorEq(oh:forward(num), output3, 0.000001, "OneHot forward number err")

   local gradInput = oh:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, input:double():zero(), 0.000001, "OneHot backward batch err")

   if pcall(function() require 'cunn' end) then
      oh:cuda()

      -- test with long input
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot forward batch long-cuda err")

      -- test with cuda input
      local input = input:cuda()
      gradOutput = gradOutput:cuda()

      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot forward batch cuda err")

      local gradInput2 = oh:backward(input, gradOutput)
      mytester:assertTensorEq(gradInput, gradInput2:double(), 0.000001, "OneHot backward batch err")
      cutorch.synchronize()

      -- non-batch mode (number input)
      mytester:assertTensorEq(oh:forward(num), output3:cuda(), 0.000001, "OneHot forward number err")
   end

   -- multi-dimensional input
   local inputSize = 2
   local input = torch.LongTensor(batchSize, inputSize):random(1, nClass)
   local gradOutput = torch.randn(batchSize, inputSize, nClass)

   local oh = nn.OneHot(nClass, 2)

   local output = oh:forward(input)
   local output2 = torch.Tensor(batchSize*inputSize, nClass):zero()
   local eye = torch.eye(nClass)
   output2:index(eye, 1, input:view(-1))
   output2:resize(batchSize, inputSize, nClass)
   mytester:assertTensorEq(output, output2, 0.000001, "OneHot 2d forward batch err")
   mytester:assert(output:dim() == 3)

   local gradInput = oh:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, input:double():zero(), 0.000001, "OneHot 2d backward batch err")

   if pcall(function() require 'cunn' end) then
      oh:cuda()

      -- test with long input
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot 2d forward batch long-cuda err")

      -- test with cuda input
      local input = input:cuda()
      gradOutput = gradOutput:cuda()

      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot 2d forward batch cuda err")

      local gradInput2 = oh:backward(input, gradOutput)
      mytester:assertTensorEq(gradInput, gradInput2:double(), 0.000001, "OneHot 2d backward batch err")

      local benchmark = false
      if benchmark then
         local input = torch.FloatTensor(50, 50):random(1,65):cuda()

         local oh = nn.OneHot(65):cuda()

         oh:forward(input)
         cutorch.synchronize()
         local a = torch.Timer()
         for i=1,10 do
            oh:forward(input)
         end
         cutorch.synchronize()
         local gputime = a:time().real

         oh:float()
         input = input:float()
         oh:forward(input)
         a = torch.Timer()
         for i=1,10 do
            oh:forward(input)
         end
         local cputime = a:time().real
         print("Onehot GPU vs CPU time", gputime, cputime)
      end
   end
end

function rnntest.NCE_main()
   local batchsize = 4
   local k = 10
   local inputsize = 3
   local outputsize = 100

   local noise = torch.Tensor(outputsize):random(1,100)

   local ncem = nn.NCEModule(inputsize, outputsize, k, noise)
   ncem.batchnoise = false
   local ncec = nn.NCECriterion()

   local input = torch.randn(batchsize, inputsize)
   local target = torch.LongTensor(batchsize):random(1,outputsize)
   local inputTable = {input, target}

   -- test training

   -- NCEModule.forward
   local output = ncem:forward(inputTable)

   mytester:assert(torch.type(output) == 'table')
   mytester:assert(#output == 4)

   local Pmt, Pms, Pnt, Pns = unpack(output)

   mytester:assertTableEq(Pmt:size():totable(), {batchsize}, 0.0000001)
   mytester:assertTableEq(Pms:size():totable(), {batchsize, k}, 0.0000001)
   mytester:assertTableEq(Pnt:size():totable(), {batchsize}, 0.0000001)
   mytester:assertTableEq(Pns:size():totable(), {batchsize, k}, 0.0000001)

   mytester:assert(ncem.sampleidx:min() >= 1 and ncem.sampleidx:max() <= outputsize)

   local sampleprob2 = noise:index(1, ncem.sampleidx:view(-1)):view(batchsize, k+1)
   mytester:assertTensorEq(sampleprob2:select(2,1), Pnt, 0.0000001)
   mytester:assertTensorEq(sampleprob2:narrow(2,2,k), Pns, 0.0000001)

   local linear = nn.Linear(inputsize, outputsize)
   linear.weight:copy(ncem.weight)
   linear.bias:copy(ncem.bias)
   local mlp = nn.Sequential():add(linear):add(nn.Exp()):add(nn.MulConstant(1/ncem.Z[1]))

   local output2_ = mlp:forward(input)
   local output2 = torch.Tensor(batchsize, k+1)
   for i=1,batchsize do
      output2[i]:index(output2_[i],1,ncem.sampleidx[i])
   end
   local Pmt2 = output2:select(2,1)
   local Pms2 = output2:narrow(2,2,k)

   mytester:assertTensorEq(Pmt, Pmt2, 0.000001)
   mytester:assertTensorEq(Pms, Pms2, 0.000001)

   -- NCECriterion.forward
   local loss = ncec:forward(output, target)

   -- eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
   local Pom = Pmt:clone()
   local mdiv = Pmt:clone():add(k, Pnt):add(0.0000001)
   Pom:cdiv(mdiv)

   -- eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
   local Pon = Pns:clone():mul(k)
   local ndiv = Pms:clone():add(k, Pns):add(0.0000001)
   Pon:cdiv(ndiv)

   -- equation 6 in ref. A

   local lossm = torch.log(Pom):sum()
   local lossn = torch.log(Pon):sum()

   local loss2 = - (lossm + lossn)/batchsize

   mytester:assert(math.abs(loss - loss2) < 0.000001)

   -- NCECriterion.backward
   local gradOutput = ncec:backward(output, target)

   mytester:assert(#gradOutput == 4)
   mytester:assert(math.abs(gradOutput[3]:sum()) < 0.0000001)
   mytester:assert(math.abs(gradOutput[4]:sum()) < 0.0000001)

   local dPmt, dPms = gradOutput[1], gradOutput[2]

   -- d Pmt / d input = -k*Pnt / ( Pmt * (Pmt + k*Pnt) )
   local dPmt2 = torch.mul(Pnt, -k):cdiv(mdiv):cdiv(torch.add(Pmt, 0.0000001)):div(batchsize)
   -- d Pms / d input = Pms / ( Pms * (Pms + k*Pns) )
   local dPms2 = Pms:clone():cdiv(ndiv):cdiv(torch.add(Pms, 0.0000001)):div(batchsize)

   mytester:assertTensorEq(dPmt, dPmt2, 0.0000001)
   mytester:assertTensorEq(dPms, dPms2, 0.0000001)

   mytester:assert(dPmt:sum() == dPmt:sum())
   mytester:assert(dPms:sum() == dPms:sum())

   -- NCEModule.backward
   ncem:zeroGradParameters()
   local gradInput = ncem:backward(inputTable, gradOutput)

   -- updateGradInput
   local gradOutput2_ = torch.zeros(batchsize, k+1)
   gradOutput2_:select(2,1):copy(gradOutput[1])
   gradOutput2_:narrow(2,2,k):copy(gradOutput[2])
   local gradOutput2 = torch.zeros(batchsize, outputsize)
   for i=1,batchsize do
      gradOutput2[i]:indexAdd(1, ncem.sampleidx[i], gradOutput2_[i])
   end
   mlp:zeroGradParameters()
   local gradInput2 = mlp:backward(input, gradOutput2)
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.0000001)

   -- accGradParameters

   local params, gradParams = ncem:parameters()
   local params2, gradParams2 = mlp:parameters()

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
   end


   if pcall(function() require 'cunn' end) then
      -- test training with cuda

      ncem:cuda()
      ncec:cuda()

      local input = input:cuda()
      local target = target:cuda()

      local inputTable = {input, target}

      -- NCEModule.forward
      local output = ncem:forward(inputTable)

      mytester:assert(torch.type(output) == 'table')
      mytester:assert(#output == 4)

      local Pmt, Pms, Pnt, Pns = unpack(output)

      mytester:assertTableEq(Pmt:size():totable(), {batchsize}, 0.0000001)
      mytester:assertTableEq(Pms:size():totable(), {batchsize, k}, 0.0000001)
      mytester:assertTableEq(Pnt:size():totable(), {batchsize}, 0.0000001)
      mytester:assertTableEq(Pns:size():totable(), {batchsize, k}, 0.0000001)

      mytester:assert(ncem.sampleidx:min() >= 1 and ncem.sampleidx:max() <= outputsize)

      local sampleprob2 = noise:cuda():index(1, ncem.sampleidx:view(-1)):view(batchsize, k+1)

      mytester:assertTensorEq(sampleprob2:select(2,1), Pnt, 0.0000001)
      mytester:assertTensorEq(sampleprob2:narrow(2,2,k), Pns, 0.0000001)

      local linear = nn.Linear(inputsize, outputsize)
      linear.weight:copy(ncem.weight)
      linear.bias:copy(ncem.bias)
      local mlp = nn.Sequential():add(linear):add(nn.Exp()):add(nn.MulConstant(1/ncem.Z[1]))
      mlp:cuda()

      local output2_ = mlp:forward(input)
      local output2 = torch.CudaTensor(batchsize, k+1)
      for i=1,batchsize do
         output2[i]:index(output2_[i],1,ncem.sampleidx[i])
      end
      local Pmt2 = output2:select(2,1)
      local Pms2 = output2:narrow(2,2,k)

      mytester:assertTensorEq(Pmt, Pmt2, 0.000001)
      mytester:assertTensorEq(Pms, Pms2, 0.000001)

      -- NCECriterion.forward
      local loss = ncec:forward(output, target)

      -- eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
      local Pom = Pmt:clone()
      local mdiv = Pmt:clone():add(k, Pnt):add(0.0000001)
      Pom:cdiv(mdiv)

      -- eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
      local Pon = Pns:clone():mul(k)
      local ndiv = Pms:clone():add(k, Pns):add(0.0000001)
      Pon:cdiv(ndiv)

      -- equation 6 in ref. A

      local lossm = torch.log(Pom):sum()
      local lossn = torch.log(Pon):sum()

      local loss2 = - (lossm + lossn)/batchsize

      mytester:assert(math.abs(loss - loss2) < 0.000001)

      -- NCECriterion.backward
      local gradOutput = ncec:backward(output, target)

      mytester:assert(#gradOutput == 4)
      mytester:assert(math.abs(gradOutput[3]:sum()) < 0.0000001)
      mytester:assert(math.abs(gradOutput[4]:sum()) < 0.0000001)

      local dPmt, dPms = gradOutput[1], gradOutput[2]

      -- d Pmt / d input = -k*Pnt / ( Pmt * (Pmt + k*Pnt) )
      local dPmt2 = torch.mul(Pnt, -k):cdiv(mdiv):cdiv(torch.add(Pmt, 0.0000001)):div(batchsize)
      -- d Pms / d input = Pms / ( Pms * (Pms + k*Pns) )
      local dPms2 = Pms:clone():cdiv(ndiv):cdiv(torch.add(Pms, 0.0000001)):div(batchsize)

      mytester:assertTensorEq(dPmt, dPmt2, 0.0000001)
      mytester:assertTensorEq(dPms, dPms2, 0.0000001)

      mytester:assert(dPmt:sum() == dPmt:sum())
      mytester:assert(dPms:sum() == dPms:sum())

      -- NCEModule.backward
      ncem:zeroGradParameters()
      local gradInput = ncem:backward(inputTable, gradOutput)

      -- updateGradInput
      local gradOutput2_ = torch.zeros(batchsize, k+1):cuda()
      gradOutput2_:select(2,1):copy(gradOutput[1])
      gradOutput2_:narrow(2,2,k):copy(gradOutput[2])
      local gradOutput2 = torch.zeros(batchsize, outputsize):cuda()
      for i=1,batchsize do
         gradOutput2[i]:indexAdd(1, ncem.sampleidx[i], gradOutput2_[i])
      end
      mlp:zeroGradParameters()
      local gradInput2 = mlp:backward(input, gradOutput2)
      mytester:assertTensorEq(gradInput[1], gradInput2, 0.0000001)

      -- accGradParameters

      local params, gradParams = ncem:parameters()
      local params2, gradParams2 = mlp:parameters()

      for i=1,#params do
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001)
      end
   end
end

function rnntest.NCE_multinomial()
   local probs = torch.Tensor(10):uniform(0,1)
   probs:div(probs:sum())
   local nce = nn.NCEModule(10, 10, 2500, probs)

   local output = torch.LongTensor()
   nce:noiseSample(output, 1000, 2500)

   local counts = torch.Tensor(10):zero()
   output:apply(function(x)
      counts[x] = counts[x] + 1
   end)

   counts:div(counts:sum())

   mytester:assertTensorEq(probs, counts, 0.001)
end

function rnntest.NCE_batchnoise()
   local batchsize = 4
   local k = 10
   local inputsize = 3
   local outputsize = 100

   local noise = torch.Tensor(outputsize):random(1,100)

   local ncem = nn.NCEModule(inputsize, outputsize, k, noise, 1)
   assert(ncem.batchnoise)
   local ncec = nn.NCECriterion()

   local ncem2 = ncem:clone()
   ncem2.batchnoise = false
   local ncec2 = ncec:clone()

   local input = torch.randn(batchsize, inputsize)
   local target = torch.LongTensor(batchsize):random(1,outputsize)
   local inputTable = {input, target}

   -- test training

   -- NCEModule.forward
   local output = ncem:forward(inputTable)

   mytester:assert(torch.type(output) == 'table')
   mytester:assert(#output == 4)

   local Pmt, Pms, Pnt, Pns = unpack(output)

   mytester:assertTableEq(Pmt:size():totable(), {batchsize}, 0.0000001)
   mytester:assertTableEq(Pms:size():totable(), {batchsize, k}, 0.0000001)
   mytester:assertTableEq(Pnt:size():totable(), {batchsize}, 0.0000001)
   mytester:assertTableEq(Pns:size():totable(), {batchsize, k}, 0.0000001)

   mytester:assert(ncem.sampleidx:min() >= 1 and ncem.sampleidx:max() <= outputsize)

   local sampleprob2 = noise:index(1, ncem.sampleidx:view(-1))
   mytester:assertTensorEq(sampleprob2:narrow(1,k+1,batchsize), Pnt, 0.0000001)
   mytester:assertTensorEq(sampleprob2:narrow(1,1,k):contiguous():view(1,k):expand(batchsize, k), Pns, 0.0000001)

   function ncem2.noiseSample(self, sampleidx, batchsize, k)
      sampleidx:resize(batchsize, k):copy(ncem.sampleidx:narrow(1,1,k):view(1, k):expand(batchsize, k))
      return sampleidx
   end

   local output2 = ncem2:forward(inputTable)
   local Pmt2, Pms2, Pnt2, Pns2 = unpack(output2)

   mytester:assertTensorEq(Pmt, Pmt2, 0.000001)
   mytester:assertTensorEq(Pms, Pms2, 0.000001)

   -- NCECriterion.forward
   local loss = ncec:forward(output, target)
   local loss2 = ncec2:forward(output, target)

   mytester:assert(math.abs(loss - loss2) < 0.000001)

   -- NCECriterion.backward
   local gradOutput = ncec:backward(output, target)
   local gradOutput2 = ncec2:backward(output, target)

   mytester:assert(#gradOutput == 4)
   mytester:assert(math.abs(gradOutput[3]:sum()) < 0.0000001)
   mytester:assert(math.abs(gradOutput[4]:sum()) < 0.0000001)

   mytester:assertTensorEq(gradOutput[1], gradOutput2[1], 0.0000001)
   mytester:assertTensorEq(gradOutput[2], gradOutput2[2], 0.0000001)

   -- NCEModule.backward
   ncem:zeroGradParameters()
   local gradInput = ncem:backward(inputTable, gradOutput)

   ncem2:zeroGradParameters()
   local gradInput2 = ncem2:backward(inputTable, gradOutput2)

   -- updateGradInput
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.0000001)

   -- accGradParameters
   local params, gradParams = ncem:parameters()
   local params2, gradParams2 = ncem2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, tostring(gradParams[i])..tostring(gradParams2[i]))
   end


   if pcall(function() require 'cunn' end) then
      -- test training with cuda

      ncem:cuda()
      ncec:cuda()

      ncem2:cuda()
      ncec2:cuda()

      local input = input:cuda()
      local target = target:cuda()
      local noise = noise:cuda()

      local inputTable = {input, target}

      -- NCEModule.forward
      local output = ncem:forward(inputTable)

      mytester:assert(torch.type(output) == 'table')
      mytester:assert(#output == 4)

      local Pmt, Pms, Pnt, Pns = unpack(output)

      mytester:assertTableEq(Pmt:size():totable(), {batchsize}, 0.0000001)
      mytester:assertTableEq(Pms:size():totable(), {batchsize, k}, 0.0000001)
      mytester:assertTableEq(Pnt:size():totable(), {batchsize}, 0.0000001)
      mytester:assertTableEq(Pns:size():totable(), {batchsize, k}, 0.0000001)

      mytester:assert(ncem.sampleidx:min() >= 1 and ncem.sampleidx:max() <= outputsize)

      local sampleprob2 = noise:index(1, ncem.sampleidx:view(-1))
      mytester:assertTensorEq(sampleprob2:narrow(1,k+1,batchsize), Pnt, 0.0000001)
      mytester:assertTensorEq(sampleprob2:narrow(1,1,k):contiguous():view(1,k):expand(batchsize, k), Pns, 0.0000001)

      function ncem2.noiseSample(self, sampleidx, batchsize, k)
         sampleidx:resize(batchsize, k):copy(ncem.sampleidx:narrow(1,1,k):view(1, k):expand(batchsize, k))
         return sampleidx
      end

      local output2 = ncem2:forward(inputTable)
      local Pmt2, Pms2, Pnt2, Pns2 = unpack(output2)

      mytester:assertTensorEq(Pmt, Pmt2, 0.000001)
      mytester:assertTensorEq(Pms, Pms2, 0.000001)

      -- NCECriterion.forward
      local loss = ncec:forward(output, target)
      local loss2 = ncec2:forward(output, target)

      mytester:assert(math.abs(loss - loss2) < 0.000001)

      -- NCECriterion.backward
      local gradOutput = ncec:backward(output, target)
      local gradOutput2 = ncec2:backward(output, target)

      mytester:assert(#gradOutput == 4)
      mytester:assert(math.abs(gradOutput[3]:sum()) < 0.0000001)
      mytester:assert(math.abs(gradOutput[4]:sum()) < 0.0000001)

      mytester:assertTensorEq(gradOutput[1], gradOutput2[1], 0.0000001)
      mytester:assertTensorEq(gradOutput[2], gradOutput2[2], 0.0000001)

      -- NCEModule.backward
      ncem:zeroGradParameters()
      local gradInput = ncem:backward(inputTable, gradOutput)

      ncem2:zeroGradParameters()
      local gradInput2 = ncem2:backward(inputTable, gradOutput2)

      -- updateGradInput
      mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.0000001)

      -- accGradParameters
      local params, gradParams = ncem:parameters()
      local params2, gradParams2 = ncem2:parameters()

      for i=1,#params do
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, tostring(gradParams[i])..tostring(gradParams2[i]))
      end
   end
end

function rnntest.NCE_multicuda()
   if not pcall(function() require 'torchx' end) then
      return
   end
   if not pcall(function() require 'cunn' end) then
      return
   end
   if cutorch.getDeviceCount() < 2 then
      return
   end
   assert(torchx.version and torchx.version >= 1, "Update torchx")

   local nclass = 1000
   local hiddensize = 20
   local batchsize = 5
   local k = 25
   local unigrams = torch.Tensor(nclass):uniform(0,1)
   local noise = torch.LongTensor(batchsize, k):random(1,nclass)

   local crit = nn.NCECriterion():cuda()
   local crit2 = nn.NCECriterion():cuda()

   local nce = nn.NCEModule(hiddensize, nclass, k, unigrams)
   nce.batchnoise = math.random() < 0.5

   -- make it deterministic
   nce.noiseSample = function(self, sampleidx, batchsize, k)
      sampleidx:resize(batchsize, k)
      sampleidx:copy(noise:narrow(1,1,batchsize))
      return sampleidx
   end

   local nce2 = nce:clone()
   nce2:cuda()

   local input = torch.randn(batchsize, hiddensize):cuda()
   local target = torch.LongTensor(batchsize):random(1,nclass):cuda()

   nce:multicuda(1, 2)

   local output = nce:forward{input, target}
   local loss = crit:forward(output, target)
   local gradOutput = crit:backward(output, target)
   nce:zeroGradParameters()
   local gradInput = nce:backward({input, target}, gradOutput)

   local output2 = nce2:forward{input, target}
   local loss2 = crit2:forward(output2, target)
   local gradOutput2 = crit2:backward(output2, target)
   nce2:zeroGradParameters()
   local gradInput2 = nce2:backward({input, target}, gradOutput2)

   mytester:assertTensorEq(output[1], output2[1], 0.00001)
   mytester:assertTensorEq(output[2], output2[2], 0.00001)
   mytester:assertTensorEq(output[3], output2[3], 0.00001)
   mytester:assertTensorEq(output[4], output2[4], 0.00001)

   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.00001)
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.00001)


   nce2:updateParameters(0.1)
   nce:updateParameters(0.1)

   mytester:assertTensorEq(nce2.bias, nce.bias, 0.000001)
   mytester:assertTensorEq(nce2.gradBias, nce.gradBias, 0.000001)
   mytester:assertTensorEq(nce2.weight[{{},{1,hiddensize/2}}]:float(), nce.weight.tensors[1]:float(), 0.000001)
   mytester:assertTensorEq(nce2.weight[{{},{1+(hiddensize/2), hiddensize}}]:float(), nce.weight.tensors[2]:float(), 0.000001)
   mytester:assertTensorEq(nce2.gradWeight[{{},{1,hiddensize/2}}]:float(), nce.gradWeight.tensors[1]:float(), 0.000001)
   mytester:assertTensorEq(nce2.gradWeight[{{},{1+(hiddensize/2), hiddensize}}]:float(), nce.gradWeight.tensors[2]:float(), 0.000001)

   -- test momentum
   nce2:updateGradParameters(0.9)
   nce:updateGradParameters(0.9)

   mytester:assertTensorEq(nce2.gradBias, nce.gradBias, 0.000001)
   mytester:assertTensorEq(nce2.momGradParams[1][{{},{1,hiddensize/2}}]:float(), nce.momGradParams[1].tensors[1]:float(), 0.000001)
   mytester:assertTensorEq(nce2.momGradParams[1][{{},{1+(hiddensize/2), hiddensize}}]:float(), nce.momGradParams[1].tensors[2]:float(), 0.000001)
   mytester:assertTensorEq(nce2.gradWeight[{{},{1,hiddensize/2}}]:float(), nce.gradWeight.tensors[1]:float(), 0.000001)
   mytester:assertTensorEq(nce2.gradWeight[{{},{1+(hiddensize/2), hiddensize}}]:float(), nce.gradWeight.tensors[2]:float(), 0.000001)
end

function rnn.test(tests, exclude, benchmark_)
   benchmark = benchmark_
   mytester = torch.Tester()
   mytester:add(rnntest)
   math.randomseed(os.time())
   if exclude then
      local excludes = {}
      assert(tests)
      tests = torch.type(tests) == 'table' and tests or {tests}
      for i,test in ipairs(tests) do
         assert(torch.type(test) == 'string')
         excludes[test] = true
      end
      tests = {}
      for testname, testfunc in pairs(rnntest.__tests) do
         if not excludes[testname] then
            table.insert(tests, testname)
         else
            print("excluding test: "..testname)
         end
      end
   end
   mytester:run(tests)
end
