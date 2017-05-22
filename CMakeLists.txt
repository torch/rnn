SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)
SET(BUILD_STATIC YES) # makes sure static targets are enabled in ADD_TORCH_PACKAGE
SET(CMAKE_C_FLAGS
  "--std=c99 -pedantic -Werror -Wall -Wextra -Wno-unused-function -D_GNU_SOURCE ${CMAKE_C_FLAGS}")

SET(src
   src/rnn.c
   )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(luasrc
  init.lua
  AbstractRecurrent.lua
  AbstractSequencer.lua
  AbstractSequencerCriterion.lua
  BiSequencer.lua
  CopyGrad.lua
  Dropout.lua
  ExpandAs.lua
  LookupTableMaskZero.lua
  MaskZero.lua
  MaskZeroCriterion.lua
  Module.lua
  Mufuru.lua
  NormStabilizer.lua
  Recurrence.lua
  RecurrentAttention.lua
  Recursor.lua
  Repeater.lua
  RepeaterCriterion.lua
  SAdd.lua
  SeqBGRU.lua
  SeqBLSTM.lua
  SeqGRU.lua
  SeqLSTM.lua
  Sequencer.lua
  SequencerCriterion.lua
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
  CategoricalEntropy.lua
  Container.lua
  Criterion.lua
  LookupTable.lua
  NCECriterion.lua
  NCEModule.lua
  ParallelTable.lua
  Reinforce.lua
  ReinforceBernoulli.lua
  ReinforceCategorical.lua
  ReinforceGamma.lua
  ReinforceNormal.lua
  ReverseSequence.lua
  Sequential.lua
  SpatialGlimpse.lua
  TotalDropout.lua
  VRClassReward.lua
  ReverseUnreverse.lua
  deprecated/SeqLSTMP.lua
  deprecated/SeqReverseSequence.lua
  deprecated/BiSequencerLM.lua
  deprecated/FastLSTM.lua
  deprecated/GRU.lua
  deprecated/LSTM.lua
)

ADD_TORCH_PACKAGE(rnn "${src}" "${luasrc}" "An RNN library for Torch")

TARGET_LINK_LIBRARIES(rnn luaT TH)

IF (BUILD_STATIC OR "$ENV{STATIC_TH}" STREQUAL "YES")
  SET_TARGET_PROPERTIES(rnn_static PROPERTIES COMPILE_FLAGS "-fPIC -DSTATIC_TH")
ENDIF()

INSTALL(FILES ${luasrc} DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/rnn")
