SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

SET(BUILD_STATIC YES) # makes sure static targets are enabled in ADD_TORCH_PACKAGE

SET(CMAKE_C_FLAGS "--std=c99 -pedantic -Werror -Wall -Wextra -Wno-unused-function -D_GNU_SOURCE ${CMAKE_C_FLAGS}")
SET(src
   init.c
)
SET(luasrc
  init.lua
  AbstractRecurrent.lua
  AbstractSequencer.lua
  AbstractSequencerCriterion.lua
  BiSequencer.lua
  BiSequencerLM.lua
  CopyGrad.lua
  Dropout.lua
  ExpandAs.lua
  deprecated/FastLSTM.lua
  deprecated/GRU.lua
  LinearNoBias.lua
  LookupTableMaskZero.lua
  deprecated/LSTM.lua
  MaskZero.lua
  MaskZeroCriterion.lua
  Module.lua
  Mufuru.lua
  NormStabilizer.lua
  Padding.lua
  Recurrence.lua
  RecurrentAttention.lua
  Recursor.lua
  Repeater.lua
  RepeaterCriterion.lua
  SAdd.lua
  SeqBRNN.lua
  SeqGRU.lua
  SeqLSTM.lua
  deprecated/SeqLSTMP.lua
  SeqReverseSequence.lua
  Sequencer.lua
  SequencerCriterion.lua
  ZeroGrad.lua
  test/bigtest.lua
  test/test.lua
  VariableLength.lua
  StepLSTM.lua
  RecLSTM.lua
  StepGRU.lua
  RecGRU.lua
  utils.lua
  LinearRNN.lua
  LookupRNN.lua
  ArgMax.lua
  BatchNormalization.lua
  BinaryClassReward.lua
  BinaryLogisticRegression.lua
  CAddTensorTable.lua
  CategoricalEntropy.lua
  Clip.lua
  Collapse.lua
  Constant.lua
  Container.lua
  Convert.lua
  Criterion.lua
  Dictionary.lua
  FireModule.lua
  Inception.lua
  Kmeans.lua
  LookupTable.lua
  ModuleCriterion.lua
  NCECriterion.lua
  NCEModule.lua
  OneHot.lua
  PCAColorTransform.lua
  ParallelTable.lua
  PrintSize.lua
  Reinforce.lua
  ReinforceBernoulli.lua
  ReinforceCategorical.lua
  ReinforceGamma.lua
  ReinforceNormal.lua
  ReverseTable.lua
  Sequential.lua
  Serial.lua
  SimpleColorTransform.lua
  SpatialBatchNormalization.lua
  SpatialBinaryConvolution.lua
  SpatialBinaryLogisticRegression.lua
  SpatialConvolution.lua
  SpatialConvolutionMM.lua
  SpatialFeatNormalization.lua
  SpatialGlimpse.lua
  SpatialMaxPooling.lua
  SpatialRegionDropout.lua
  SpatialUniformCrop.lua
  TotalDropout.lua
  VRClassReward.lua
  WhiteNoise.lua
  ZipTable.lua
  ZipTableOneToMany.lua
)

ADD_TORCH_PACKAGE(rnn "${src}" "${luasrc}" "An RNN library for Torch")

TARGET_LINK_LIBRARIES(rnn luaT TH)

SET_TARGET_PROPERTIES(rnn_static PROPERTIES COMPILE_FLAGS "-fPIC -DSTATIC_TH")

INSTALL(FILES ${luasrc} DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/rnn")
