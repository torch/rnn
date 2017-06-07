#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda/StepGRU.cu"
#else

#if defined(THC_REAL_IS_HALF)
#define _REAL(val) THC_float2half(val)
#else
#define _REAL(val) (val)
#endif

static int nn_(StepGRU_updateOutput)(lua_State *L) {
  THCState *state = getCudaState(L);
  THCTensor *weight = (THCTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *bias = (THCTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THCTensor *gates = (THCTensor *)luaT_checkudata(L, 3, torch_Tensor);
  THCTensor *cur_x = (THCTensor *)luaT_checkudata(L, 4, torch_Tensor);
  THCTensor *prev_h = (THCTensor *)luaT_checkudata(L, 5, torch_Tensor);
  int inputsize = luaL_checkinteger(L, 6);
  int outputsize = luaL_checkinteger(L, 7);
  THCTensor *next_h = (THCTensor *)luaT_checkudata(L, 8, torch_Tensor);

  int batchsize = THCTensor_(size)(state, cur_x, 0);
  if (THCTensor_(size)(state, cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");

  THLongStorage* size = THLongStorage_newWithSize2(1, 3 * outputsize);
  THCTensor *buffer = THCTensor_(newView)(state, bias, size);
  buffer->stride[0] = 0;
  buffer->size[0] = batchsize;

  THCTensor_(resize2d)(state, next_h, batchsize, outputsize);
  long nElement = THCTensor_(nElement)(state, gates);
  THCTensor_(resize2d)(state, gates, batchsize, 3 * outputsize);
  if (nElement != batchsize * 3 * outputsize)
    THCTensor_(fill)(state, gates, _REAL(0));

  THCTensor *Wx = THCTensor_(newNarrow)(state, weight, 0, 0, inputsize);
  THCTensor *Wh = THCTensor_(newNarrow)(state, weight, 0, inputsize, outputsize);
  THCTensor *sub_gates = THCTensor_(newNarrow)(state, gates, 1, 0, 2 * outputsize);
  THCTensor *sub_Wh = THCTensor_(newNarrow)(state, Wh, 1, 0, 2 * outputsize);
  // r = sig(Wx * x + Wh * prev_h + b)
  THCTensor *reset_gate = THCTensor_(newNarrow)(state, gates, 1, 0, outputsize);
  // u = sig(Wx * x + Wh * prev_h + b)
  THCTensor *update_gate = THCTensor_(newNarrow)(state, gates, 1, outputsize, outputsize);
  // hc = tanh(Wx * x + Wh * r . prev_h + b)
  THCTensor *hidden_candidate = THCTensor_(newNarrow)(state, gates, 1, 2*outputsize, outputsize);

  // forward
  THCTensor_(addmm)(state, gates, _REAL(1), buffer, _REAL(1), cur_x, Wx);
  THCTensor_(addmm)(state, sub_gates, _REAL(1), sub_gates, _REAL(1), prev_h, sub_Wh);
  THCTensor_(sigmoid)(state, sub_gates, sub_gates);

  // temporary buffer : r . prev_h
  THCTensor_(cmul)(state, next_h, reset_gate, prev_h);
  THCTensor_(narrow)(state, sub_Wh, Wh, 1, 2 * outputsize, outputsize);
  // hc += Wh * r . prev_h
  THCTensor_(addmm)(state, hidden_candidate, _REAL(1), hidden_candidate, _REAL(1), next_h, sub_Wh);
  // hc = tanh(Wx * x + Wh * r . prev_h + b)
  THCTensor_(tanh)(state, hidden_candidate, hidden_candidate);
  // (1-u) . hc = hc - (u . hc)
  THCTensor_(addcmul)(state, next_h, hidden_candidate, _REAL(-1), update_gate, hidden_candidate);
  // next_h = (1-u) . hc + u . prev_h
  THCTensor_(addcmul)(state, next_h, next_h, _REAL(1), update_gate, prev_h);

  THCTensor_(free)(state, Wx);
  THCTensor_(free)(state, Wh);
  THCTensor_(free)(state, buffer);
  THCTensor_(free)(state, reset_gate);
  THCTensor_(free)(state, update_gate);
  THCTensor_(free)(state, hidden_candidate);
  THCTensor_(free)(state, sub_gates);
  THCTensor_(free)(state, sub_Wh);
  THLongStorage_free(size);

  return 1;
}

static int nn_(StepGRU_backward)(lua_State *L) {
  THCState *state = getCudaState(L);
  THCTensor *weight = (THCTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *gates = (THCTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THCTensor *gradWeight = (THCTensor *)luaT_checkudata(L, 3, torch_Tensor);
  THCTensor *grad_b = (THCTensor *)luaT_checkudata(L, 4, torch_Tensor);
  THCTensor *grad_gates = (THCTensor *)luaT_checkudata(L, 5, torch_Tensor);
  THCTensor *buffer = (THCTensor *)luaT_checkudata(L, 6, torch_Tensor);
  THCTensor *cur_x = (THCTensor *)luaT_checkudata(L, 7, torch_Tensor);
  THCTensor *prev_h = (THCTensor *)luaT_checkudata(L, 8, torch_Tensor);
  THCTensor *grad_next_h = (THCTensor *)luaT_checkudata(L, 9, torch_Tensor);
  lua_Number scale = luaL_checknumber(L, 10);
  int inputsize = luaL_checkinteger(L, 11);
  int outputsize = luaL_checkinteger(L, 12);
  THCTensor *grad_cur_x = (THCTensor *)luaT_checkudata(L, 13, torch_Tensor);
  THCTensor *grad_prev_h = (THCTensor *)luaT_checkudata(L, 14, torch_Tensor);

  int batchsize = THCTensor_(size)(state, cur_x, 0);
  if (THCTensor_(size)(state, cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");
  if (THCTensor_(size)(state, grad_next_h, 1) != outputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected gradOutput[1]:size(2) == outputsize");

  THCTensor_(resize2d)(state, grad_cur_x, batchsize, inputsize);
  THCTensor_(resize2d)(state, grad_prev_h, batchsize, outputsize);
  THCTensor_(resize2d)(state, grad_gates, batchsize, 3 * outputsize);

  THCTensor *Wx = THCTensor_(newNarrow)(state, weight, 0, 0, inputsize);
  THCTensor *Wh = THCTensor_(newNarrow)(state, weight, 0, inputsize, outputsize);
  THCTensor *reset_gate = THCTensor_(newNarrow)(state, gates, 1, 0, outputsize);
  THCTensor *update_gate = THCTensor_(newNarrow)(state, gates, 1, outputsize, outputsize);
  THCTensor *hidden_candidate = THCTensor_(newNarrow)(state, gates, 1, 2*outputsize, outputsize);
  THCTensor *grad_Wx = THCTensor_(newNarrow)(state, gradWeight, 0, 0, inputsize);
  THCTensor *grad_Wh = THCTensor_(newNarrow)(state, gradWeight, 0, inputsize, outputsize);
  THCTensor *grad_reset_gate = THCTensor_(newNarrow)(state, grad_gates, 1, 0, outputsize);
  THCTensor *grad_update_gate = THCTensor_(newNarrow)(state, grad_gates, 1, outputsize, outputsize);
  THCTensor *grad_hidden_candidate = THCTensor_(newNarrow)(state, grad_gates, 1, 2*outputsize, outputsize);

  THCTensor *sub_Wh = THCTensor_(newNarrow)(state, Wh, 1, 2 * outputsize, outputsize);
  THCTensor *sub_Wh_t = THCTensor_(newTranspose)(state, sub_Wh, 0, 1);
  THCTensor *Wx_t = THCTensor_(newTranspose)(state, Wx, 0, 1);
  THCTensor *cur_x_t = THCTensor_(newTranspose)(state, cur_x, 0, 1);
  THCTensor *sub_grad_gates = THCTensor_(newNarrow)(state, grad_gates, 1, 0, 2 * outputsize);
  THCTensor *sub_grad_Wh = THCTensor_(newNarrow)(state, grad_Wh, 1, 0, 2 * outputsize);
  THCTensor *prev_h_t = THCTensor_(newTranspose)(state, prev_h, 0, 1);

  // use grad_update_gate as temporary buffer to compute grad_hidden_candidate and grad_reset_gate
  THCTensor_(fill)(state, grad_update_gate, _REAL(0));
  THCTensor_(addcmul)(state, grad_update_gate, grad_next_h, _REAL(-1), update_gate, grad_next_h);
  THCTensor_(fill)(state, grad_hidden_candidate, _REAL(1));
  THCTensor_(addcmul)(state, grad_hidden_candidate, grad_hidden_candidate, _REAL(-1),
                      hidden_candidate, hidden_candidate);
  THCTensor_(cmul)(state, grad_hidden_candidate, grad_hidden_candidate, grad_update_gate);

  THCTensor_(fill)(state, grad_update_gate, _REAL(0));
  THCTensor_(addmm)(state, grad_update_gate, _REAL(1), grad_update_gate, _REAL(1),
                    grad_hidden_candidate, sub_Wh_t);
  THCTensor_(cmul)(state, grad_update_gate, grad_update_gate, prev_h);
  THCTensor_(fill)(state, grad_reset_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_reset_gate, grad_reset_gate, _REAL(-1), reset_gate);
  THCTensor_(cmul)(state, grad_reset_gate, grad_reset_gate, reset_gate);
  THCTensor_(cmul)(state, grad_reset_gate, grad_reset_gate, grad_update_gate);

  THCTensor_(cadd)(state, buffer, prev_h, _REAL(-1), hidden_candidate);
  THCTensor_(fill)(state, grad_update_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_update_gate, grad_update_gate, _REAL(-1), update_gate);
  THCTensor_(cmul)(state, grad_update_gate, grad_update_gate, update_gate);
  THCTensor_(cmul)(state, grad_update_gate, grad_update_gate, buffer);
  THCTensor_(cmul)(state, grad_update_gate, grad_update_gate, grad_next_h);
  THCTensor_(addmm)(state, grad_cur_x, _REAL(0), grad_cur_x, _REAL(1), grad_gates, Wx_t);
  THCTensor_(addmm)(state, grad_Wx, _REAL(scale), grad_Wx, _REAL(1), cur_x_t, grad_gates);
  THCTensor_(addmm)(state, sub_grad_Wh, _REAL(scale), sub_grad_Wh,
                    _REAL(1), prev_h_t, sub_grad_gates);

  THCTensor_(resize1d)(state, buffer, outputsize);
  THCTensor_(sum)(state, buffer, grad_gates, 0, 0);
  THCTensor_(cadd)(state, grad_b, grad_b, _REAL(scale), buffer);
  THCTensor_(cmul)(state, buffer, prev_h, reset_gate);

  THCTensor_(narrow)(state, sub_grad_Wh, grad_Wh, 1, 2 * outputsize, outputsize);
  THCTensor_(transpose)(state, cur_x_t, buffer, 0, 1); // reuse cur_x_t as buffer_t
  THCTensor_(addmm)(state, sub_grad_Wh, _REAL(scale),
                    sub_grad_Wh, _REAL(1), cur_x_t, grad_hidden_candidate);
  THCTensor_(cmul)(state, grad_prev_h, grad_next_h, update_gate);

  THCTensor_(narrow)(state, sub_Wh, Wh, 1, 0, 2 * outputsize);
  THCTensor_(transpose)(state, cur_x_t, sub_Wh, 0, 1); // reuse cur_x_t as sub_Wh_t
  THCTensor_(addmm)(state, grad_prev_h, _REAL(1), grad_prev_h, _REAL(1), sub_grad_gates, cur_x_t);

  THCTensor_(addmm)(state, buffer, _REAL(0), buffer, _REAL(1), grad_hidden_candidate, sub_Wh_t);
  THCTensor_(cmul)(state, buffer, buffer, reset_gate);
  THCTensor_(cadd)(state, grad_prev_h, grad_prev_h, _REAL(1), buffer);

  THCTensor_(free)(state, Wx);
  THCTensor_(free)(state, Wh);
  THCTensor_(free)(state, reset_gate);
  THCTensor_(free)(state, update_gate);
  THCTensor_(free)(state, hidden_candidate);

  THCTensor_(free)(state, grad_Wx);
  THCTensor_(free)(state, grad_Wh);
  THCTensor_(free)(state, grad_reset_gate);
  THCTensor_(free)(state, grad_update_gate);
  THCTensor_(free)(state, grad_hidden_candidate);

  THCTensor_(free)(state, sub_Wh);
  THCTensor_(free)(state, sub_Wh_t);
  THCTensor_(free)(state, Wx_t);
  THCTensor_(free)(state, cur_x_t);
  THCTensor_(free)(state, sub_grad_gates);
  THCTensor_(free)(state, sub_grad_Wh);
  THCTensor_(free)(state, prev_h_t);

  return 2;
}

static const struct luaL_Reg nn_(StepGRU__) [] = {
  {"StepGRU_updateOutput", nn_(StepGRU_updateOutput)},
  {"StepGRU_backward", nn_(StepGRU_backward)},
  {NULL, NULL}
};

static void nn_(StepGRU_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(StepGRU__), "nn");
  lua_pop(L,1);
}

#undef _REAL
#endif
