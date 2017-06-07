#include "luaT.h"
#include "TH.h"

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

#if defined(USE_CUDA)
#if defined(__cplusplus)
extern "C" {
#endif
int cuda_librnn_init(lua_State *L);
#if defined(__cplusplus)
}
#endif
#endif
