#include "rnn.h"
#include "THC/THC.h"

static THCState *getCudaState(lua_State *L)
{
    lua_getglobal(L, "cutorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THCState * state = (THCState *)lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}

#define torch_(NAME) TH_CONCAT_3(torch_, CReal, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., CReal, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, CReal, NAME)

#include "generic/cuda/VariableLength.cu"
#include "THC/THCGenerateFloatTypes.h"

#include "generic/cuda/StepLSTM.cu"
#include "THC/THCGenerateFloatTypes.h"

#include "generic/cuda/StepGRU.cu"
#include "THC/THCGenerateFloatTypes.h"

#if defined(__cplusplus)
extern "C" {
#endif

int cuda_librnn_init(lua_State *L)
{
  nn_CudaVariableLength_init(L);
  nn_CudaStepLSTM_init(L);
  nn_CudaStepGRU_init(L);

  nn_CudaDoubleVariableLength_init(L);
  nn_CudaDoubleStepLSTM_init(L);
  nn_CudaDoubleStepGRU_init(L);

#ifdef CUDA_HALF_TENSOR
  nn_CudaHalfVariableLength_init(L);
  nn_CudaHalfStepLSTM_init(L);
  nn_CudaHalfStepGRU_init(L);
#endif

  return 1;
}

#if defined(__cplusplus)
}
#endif
