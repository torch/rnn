#include "rnn.h"

#ifdef _OPENMP
#include "omp.h"
#endif

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/c/VariableLength.c"
#include "THGenerateFloatTypes.h"

#include "generic/c/StepLSTM.c"
#include "THGenerateFloatTypes.h"

#include "generic/c/StepGRU.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_librnn(lua_State *L)
{
  nn_FloatVariableLength_init(L);
  nn_FloatStepLSTM_init(L);
  nn_FloatStepGRU_init(L);

  nn_DoubleVariableLength_init(L);
  nn_DoubleStepLSTM_init(L);
  nn_DoubleStepGRU_init(L);

#if defined(USE_CUDA)
  return cuda_librnn_init(L);
#else
  return 1;
#endif
}
