#include "luaT.h"
#include "TH.h"

#ifdef _OPENMP
#include "omp.h"
#endif

#include "error.h"
#include "utils.h"
#include <stdlib.h>

typedef struct {
  long length;
  long index;
  long assigned_row;
} Sample;

static int sample_compare(const void *a_, const void *b_) {
  Sample *a = (Sample*) a_;
  Sample *b = (Sample*) b_;
  return a->length < b->length ? -1 : a->length > b->length;
}

static long get_max_length(lua_State *L, int lengths_index) {
  const int current_top = lua_gettop(L);
  long max_length = 0;

  lua_pushnil(L);
  while (lua_next(L, lengths_index) != 0) {
    const int inner_index = current_top + 2;

    lua_pushnil(L);
    while (lua_next(L, inner_index) != 0) {
      long length = lua_tointeger(L, -1);
      if (length > max_length)
        max_length = length;
      lua_pop(L, 1);
    }

    lua_pop(L, 1);
  }

  return max_length;
}

static long get_n_samples(lua_State *L, int lengths_index) {
  long count = 0;

  lua_pushnil(L);
  while (lua_next(L, lengths_index) != 0) {
    count += lua_objlen(L, -1);
    lua_pop(L, 1);
  }

  return count;
}

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/VariableLength.c"
#include "THGenerateFloatTypes.h"

#include "generic/StepLSTM.c"
#include "THGenerateFloatTypes.h"

#include "generic/StepGRU.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_librnn(lua_State *L)
{
  nn_FloatVariableLength_init(L);
  nn_FloatStepLSTM_init(L);
  nn_FloatStepGRU_init(L);

  nn_DoubleVariableLength_init(L);
  nn_DoubleStepLSTM_init(L);
  nn_DoubleStepGRU_init(L);

  return 1;
}
