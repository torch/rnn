#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StepGRU.c"
#else

static int nn_(StepGRU_updateOutput)(lua_State *L) {
  THTensor *weight = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *bias = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gates = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *cur_x = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 5, torch_Tensor);
  int inputsize = luaL_checkinteger(L, 6);
  int outputsize = luaL_checkinteger(L, 7);
  THTensor *next_h = luaT_checkudata(L, 8, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");

  THLongStorage* size = THLongStorage_newWithSize2(1, 3 * outputsize);
  THTensor *buffer = THTensor_(newView)(bias, size);
  buffer->stride[0] = 0;
  buffer->size[0] = batchsize;

  THTensor_(resize2d)(next_h, batchsize, outputsize);
  THTensor_(resize2d)(gates, batchsize, 3 * outputsize);

  THTensor *Wx = THTensor_(newNarrow)(weight, 0, 0, inputsize);
  THTensor *Wh = THTensor_(newNarrow)(weight, 0, inputsize, outputsize);
  THTensor *sub_gates = THTensor_(newNarrow)(gates, 1, 0, 2 * outputsize);
  THTensor *sub_Wh = THTensor_(newNarrow)(Wh, 1, 0, 2 * outputsize);
  THTensor *reset_gate = THTensor_(newNarrow)(gates, 1, 0, outputsize); // r = sig(Wx * x + Wh * prev_h + b)
  THTensor *update_gate = THTensor_(newNarrow)(gates, 1, outputsize, outputsize); // u = sig(Wx * x + Wh * prev_h + b)
  THTensor *hidden_candidate = THTensor_(newNarrow)(gates, 1, 2*outputsize, outputsize); // hc = tanh(Wx * x + Wh * r . prev_h + b)

  //THTensor_(fill)(gates, 0);

  // forward
  THTensor_(addmm)(gates, 1, buffer, 1, cur_x, Wx);
  THTensor_(addmm)(sub_gates, 1, sub_gates, 1, prev_h, sub_Wh);
  THTensor_(sigmoid)(sub_gates, sub_gates);

  THTensor_(cmul)(next_h, reset_gate, prev_h); // temporary buffer : r . prev_h
  THTensor_(narrow)(sub_Wh, Wh, 1, 2 * outputsize, outputsize);
  THTensor_(addmm)(hidden_candidate, 1, hidden_candidate, 1, next_h, sub_Wh); // hc += Wh * r . prev_h
  THTensor_(tanh)(hidden_candidate, hidden_candidate); // hc = tanh(Wx * x + Wh * r . prev_h + b)
  THTensor_(addcmul)(next_h, hidden_candidate, -1, update_gate, hidden_candidate); // (1-u) . hc = hc - (u . hc)
  THTensor_(addcmul)(next_h, next_h, 1, update_gate, prev_h); //next_h = (1-u) . hc + u . prev_h

  THTensor_(free)(Wx);
  THTensor_(free)(Wh);
  THTensor_(free)(buffer);
  THTensor_(free)(reset_gate);
  THTensor_(free)(update_gate);
  THTensor_(free)(hidden_candidate);
  THTensor_(free)(sub_gates);
  THTensor_(free)(sub_Wh);
  THLongStorage_free(size);

  return 1;
}

static int nn_(StepGRU_backward)(lua_State *L) {
  THTensor *weight = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *gates = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradWeight = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *grad_b = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *grad_gates = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *buffer = luaT_checkudata(L, 6, torch_Tensor);
  THTensor *cur_x = luaT_checkudata(L, 7, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 8, torch_Tensor);
  THTensor *grad_next_h = luaT_checkudata(L, 9, torch_Tensor);
  lua_Number scale = luaL_checknumber(L, 10);
  int inputsize = luaL_checkinteger(L, 11);
  int outputsize = luaL_checkinteger(L, 12);
  THTensor *grad_cur_x = luaT_checkudata(L, 13, torch_Tensor);
  THTensor *grad_prev_h = luaT_checkudata(L, 14, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");
  if (THTensor_(size)(grad_next_h, 1) != outputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected gradOutput[1]:size(2) == outputsize");

  THTensor_(resize2d)(grad_cur_x, batchsize, inputsize);
  THTensor_(resize2d)(grad_prev_h, batchsize, outputsize);
  THTensor_(resize2d)(grad_gates, batchsize, 3 * outputsize);
  THTensor_(fill)(grad_gates, 0);

  THTensor *Wx = THTensor_(newNarrow)(weight, 0, 0, inputsize);
  THTensor *Wh = THTensor_(newNarrow)(weight, 0, inputsize, outputsize);
  THTensor *reset_gate = THTensor_(newNarrow)(gates, 1, 0, outputsize);
  THTensor *update_gate = THTensor_(newNarrow)(gates, 1, outputsize, outputsize);
  THTensor *hidden_candidate = THTensor_(newNarrow)(gates, 1, 2*outputsize, outputsize);
  THTensor *grad_Wx = THTensor_(newNarrow)(gradWeight, 0, 0, inputsize);
  THTensor *grad_Wh = THTensor_(newNarrow)(gradWeight, 0, inputsize, outputsize);
  THTensor *grad_reset_gate = THTensor_(newNarrow)(grad_gates, 1, 0, outputsize);
  THTensor *grad_update_gate = THTensor_(newNarrow)(grad_gates, 1, outputsize, outputsize);
  THTensor *grad_hidden_candidate = THTensor_(newNarrow)(grad_gates, 1, 2*outputsize, outputsize);

  THTensor *sub_Wh = THTensor_(newNarrow)(Wh, 1, 2 * outputsize, outputsize);
  THTensor *sub_Wh_t = THTensor_(newTranspose)(sub_Wh, 0, 1);
  THTensor *Wx_t = THTensor_(newTranspose)(Wx, 0, 1);
  THTensor *cur_x_t = THTensor_(newTranspose)(cur_x, 0, 1);
  THTensor *sub_grad_gates = THTensor_(newNarrow)(grad_gates, 1, 0, 2 * outputsize);
  THTensor *sub_grad_Wh = THTensor_(newNarrow)(grad_Wh, 1, 0, 2 * outputsize);
  THTensor *prev_h_t = THTensor_(newTranspose)(prev_h, 0, 1);

  // use grad_update_gate as temporary buffer to compute grad_hidden_candidate and grad_reset_gate
  THTensor_(fill)(grad_update_gate, 0);
  THTensor_(addcmul)(grad_update_gate, grad_next_h, -1, update_gate, grad_next_h);
  THTensor_(fill)(grad_hidden_candidate, 1);
  THTensor_(addcmul)(grad_hidden_candidate, grad_hidden_candidate, -1, hidden_candidate, hidden_candidate);
  THTensor_(cmul)(grad_hidden_candidate, grad_hidden_candidate, grad_update_gate);

  THTensor_(fill)(grad_update_gate, 0);
  THTensor_(addmm)(grad_update_gate, 1, grad_update_gate, 1, grad_hidden_candidate, sub_Wh_t);
  THTensor_(cmul)(grad_update_gate, grad_update_gate, prev_h);
  THTensor_(fill)(grad_reset_gate, 1);
  THTensor_(cadd)(grad_reset_gate, grad_reset_gate, -1, reset_gate);
  THTensor_(cmul)(grad_reset_gate, grad_reset_gate, reset_gate);
  THTensor_(cmul)(grad_reset_gate, grad_reset_gate, grad_update_gate);

  THTensor_(cadd)(buffer, prev_h, -1, hidden_candidate);
  THTensor_(fill)(grad_update_gate, 1);
  THTensor_(cadd)(grad_update_gate, grad_update_gate, -1, update_gate);
  THTensor_(cmul)(grad_update_gate, grad_update_gate, update_gate);
  THTensor_(cmul)(grad_update_gate, grad_update_gate, buffer);
  THTensor_(cmul)(grad_update_gate, grad_update_gate, grad_next_h);
  THTensor_(addmm)(grad_cur_x, 0, grad_cur_x, 1, grad_gates, Wx_t);
  THTensor_(addmm)(grad_Wx, scale, grad_Wx, 1, cur_x_t, grad_gates);
  THTensor_(addmm)(sub_grad_Wh, scale, sub_grad_Wh, 1, prev_h_t, sub_grad_gates);

  THTensor_(resize1d)(buffer, outputsize);
  THTensor_(sum)(buffer, grad_gates, 0);
  THTensor_(cadd)(grad_b, grad_b, scale, buffer);
  THTensor_(cmul)(buffer, prev_h, reset_gate);

  THTensor_(narrow)(sub_grad_Wh, grad_Wh, 1, 2 * outputsize, outputsize);
  THTensor_(transpose)(cur_x_t, buffer, 0, 1); // reuse cur_x_t as buffer_t
  THTensor_(addmm)(sub_grad_Wh, scale, sub_grad_Wh, 1, cur_x_t, grad_hidden_candidate);
  THTensor_(cmul)(grad_prev_h, grad_next_h, update_gate);

  THTensor_(narrow)(sub_Wh, Wh, 1, 0, 2 * outputsize);
  THTensor_(transpose)(cur_x_t, sub_Wh, 0, 1); // reuse cur_x_t as sub_Wh_t
  THTensor_(addmm)(grad_prev_h, 1, grad_prev_h, 1, sub_grad_gates, cur_x_t);

  THTensor_(addmm)(buffer, 0, buffer, 1, grad_hidden_candidate, sub_Wh_t);
  THTensor_(cmul)(buffer, buffer, reset_gate);
  THTensor_(cadd)(grad_prev_h, grad_prev_h, 1, buffer);

  THTensor_(free)(Wx);
  THTensor_(free)(Wh);
  THTensor_(free)(reset_gate);
  THTensor_(free)(update_gate);
  THTensor_(free)(hidden_candidate);

  THTensor_(free)(grad_Wx);
  THTensor_(free)(grad_Wh);
  THTensor_(free)(grad_reset_gate);
  THTensor_(free)(grad_update_gate);
  THTensor_(free)(grad_hidden_candidate);

  THTensor_(free)(sub_Wh);
  THTensor_(free)(sub_Wh_t);
  THTensor_(free)(Wx_t);
  THTensor_(free)(cur_x_t);
  THTensor_(free)(sub_grad_gates);
  THTensor_(free)(sub_grad_Wh);
  THTensor_(free)(prev_h_t);

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

#endif
