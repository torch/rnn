local _ = require 'moses'
local BN, parent = nn.BatchNormalization, nn.Module

-- for sharedClone
local params = _.clone(parent.dpnn_parameters)
table.insert(params, 'running_mean')
table.insert(params, 'running_var')
BN.dpnn_parameters = params
