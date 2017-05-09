#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StepLSTM.c"
#else

static int nn_(StepLSTM_updateOutput)(lua_State *L) {
  THTensor *weight = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *bias = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gates = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *cur_x = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *prev_c = luaT_checkudata(L, 6, torch_Tensor);
  int inputsize = luaL_checkinteger(L, 7);
  int hiddensize = luaL_checkinteger(L, 8);
  int outputsize = luaL_checkinteger(L, 9);
  THTensor *next_h = luaT_checkudata(L, 10, torch_Tensor); // when LSTMP pass hidden[t]
  THTensor *next_c = luaT_checkudata(L, 11, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");

  THLongStorage* size = THLongStorage_newWithSize2(1, 4 * hiddensize);
  THTensor *buffer = THTensor_(newView)(bias, size);
  buffer->stride[0] = 0;
  buffer->size[0] = batchsize;

  THTensor *Wx = THTensor_(newNarrow)(weight, 0, 0, inputsize);
  THTensor *Wh = THTensor_(newNarrow)(weight, 0, inputsize, outputsize);

  THTensor_(resize2d)(next_h, batchsize, hiddensize);
  THTensor_(resize2d)(next_c, batchsize, hiddensize);

  THTensor_(resize2d)(gates, batchsize, 4 * hiddensize);
  //THTensor_(fill)(gates, 0);

  // forward
  THTensor_(addmm)(gates, 1, buffer, 1, cur_x, Wx);
  THTensor_(addmm)(gates, 1, gates, 1, prev_h, Wh);

  THTensor_(narrow)(buffer, gates, 1, 0, 3 * hiddensize);
  THTensor_(sigmoid)(buffer, buffer);

  THTensor_(narrow)(buffer, gates, 1, 3 * hiddensize, hiddensize);
  THTensor_(tanh)(buffer, buffer);

  THTensor *input_gate = THTensor_(newNarrow)(gates, 1, 0, hiddensize);
  THTensor *forget_gate = THTensor_(newNarrow)(gates, 1, hiddensize, hiddensize);
  THTensor *output_gate = THTensor_(newNarrow)(gates, 1, 2*hiddensize, hiddensize);
  THTensor *input_transform = THTensor_(newNarrow)(gates, 1, 3*hiddensize, hiddensize);

  THTensor_(cmul)(next_h, input_gate, input_transform);
  THTensor_(cmul)(next_c, forget_gate, prev_c);
  THTensor_(cadd)(next_c, next_c, 1, next_h);
  THTensor_(tanh)(next_h, next_c);
  THTensor_(cmul)(next_h, next_h, output_gate);

  THTensor_(free)(Wx);
  THTensor_(free)(Wh);
  THTensor_(free)(buffer);
  THTensor_(free)(input_gate);
  THTensor_(free)(forget_gate);
  THTensor_(free)(output_gate);
  THTensor_(free)(input_transform);
  THLongStorage_free(size);

  if (lua_gettop(L) > 11) // implements LSTMP (P stands for projection layer)
  {
    THTensor *hidden = next_h;
    THTensor *weightO = luaT_checkudata(L, 12, torch_Tensor);
    next_h = luaT_checkudata(L, 13, torch_Tensor);
    THTensor_(resize2d)(next_h, batchsize, outputsize);
    THTensor_(addmm)(next_h, 0, next_h, 1, hidden, weightO);
    // push results onto stack
    luaT_pushudata(L, next_c, torch_Tensor);
  }

  return 2;
}

static int nn_(StepLSTM_backward)(lua_State *L) {
  THTensor *weight = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *gates = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradWeight = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *grad_b = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *grad_gates = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *grad_gates_sum = luaT_checkudata(L, 6, torch_Tensor);
  THTensor *cur_x = luaT_checkudata(L, 7, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 8, torch_Tensor);
  THTensor *prev_c = luaT_checkudata(L, 9, torch_Tensor);
  THTensor *next_c = luaT_checkudata(L, 10, torch_Tensor);
  THTensor *grad_next_h = luaT_checkudata(L, 11, torch_Tensor);
  THTensor *grad_next_c = luaT_checkudata(L, 12, torch_Tensor);
  lua_Number scale = luaL_checknumber(L, 13);
  int inputsize = luaL_checkinteger(L, 14);
  int hiddensize = luaL_checkinteger(L, 15);
  int outputsize = luaL_checkinteger(L, 16);
  THTensor *grad_cur_x = luaT_checkudata(L, 17, torch_Tensor);
  THTensor *grad_prev_h = luaT_checkudata(L, 18, torch_Tensor);
  THTensor *grad_prev_c = luaT_checkudata(L, 19, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");
  if (THTensor_(size)(grad_next_h, 1) != outputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected gradOutput[1]:size(2) == outputsize");

  if (lua_gettop(L) > 19) // LSTMP
  {
    THTensor *weightO = luaT_checkudata(L, 20, torch_Tensor);
    THTensor *hidden = luaT_checkudata(L, 21, torch_Tensor);
    THTensor *gradWeightO = luaT_checkudata(L, 22, torch_Tensor);
    THTensor *grad_hidden = luaT_checkudata(L, 23, torch_Tensor);

    THTensor *hidden_t = THTensor_(newTranspose)(hidden, 0, 1);
    THTensor *weightO_t = THTensor_(newTranspose)(weightO, 0, 1);

    THTensor_(addmm)(gradWeightO, scale, gradWeightO, 1, hidden_t, grad_next_h);
    THTensor_(resize2d)(grad_hidden, batchsize, hiddensize);
    THTensor_(addmm)(grad_hidden, 0, grad_hidden, 1, grad_next_h, weightO_t);

    grad_next_h = grad_hidden;

    THTensor_(free)(hidden_t);
    THTensor_(free)(weightO_t);

    // push results to top of stack
    luaT_pushudata(L, grad_cur_x, torch_Tensor);
    luaT_pushudata(L, grad_prev_h, torch_Tensor);
    luaT_pushudata(L, grad_prev_c, torch_Tensor);
  }

  THTensor_(resize2d)(grad_cur_x, batchsize, inputsize);
  THTensor_(resize2d)(grad_prev_h, batchsize, outputsize);
  THTensor_(resize2d)(grad_prev_c, batchsize, hiddensize);

  // these tensors were set-up in updateOutput
  THTensor *Wx = THTensor_(newNarrow)(weight, 0, 0, inputsize);
  THTensor *Wh = THTensor_(newNarrow)(weight, 0, inputsize, outputsize);

  THTensor *input_gate = THTensor_(newNarrow)(gates, 1, 0, hiddensize);
  THTensor *forget_gate = THTensor_(newNarrow)(gates, 1, hiddensize, hiddensize);
  THTensor *output_gate = THTensor_(newNarrow)(gates, 1, 2*hiddensize, hiddensize);
  THTensor *input_transform = THTensor_(newNarrow)(gates, 1, 3*hiddensize, hiddensize);

  // set-up grad tensors
  THTensor *grad_Wx = THTensor_(newNarrow)(gradWeight, 0, 0, inputsize);
  THTensor *grad_Wh = THTensor_(newNarrow)(gradWeight, 0, inputsize, outputsize);

  THTensor_(resize2d)(grad_gates, batchsize, 4 * hiddensize);
  THTensor_(fill)(grad_gates, 0);

  THTensor *grad_input_gate = THTensor_(newNarrow)(grad_gates, 1, 0, hiddensize);
  THTensor *grad_forget_gate = THTensor_(newNarrow)(grad_gates, 1, hiddensize, hiddensize);
  THTensor *grad_output_gate = THTensor_(newNarrow)(grad_gates, 1, 2*hiddensize, hiddensize);
  THTensor *grad_input_transform = THTensor_(newNarrow)(grad_gates, 1, 3*hiddensize, hiddensize);

  // backward

  // we use grad_[input,forget,output]_gate as temporary buffers to compute grad_prev_c.
  THTensor_(tanh)(grad_input_gate, next_c);
  THTensor_(cmul)(grad_forget_gate, grad_input_gate, grad_input_gate);

  THTensor_(fill)(grad_output_gate, 1);
  THTensor_(cadd)(grad_output_gate, grad_output_gate, -1, grad_forget_gate);
  THTensor_(cmul)(grad_output_gate, grad_output_gate, output_gate);
  THTensor_(cmul)(grad_output_gate, grad_output_gate, grad_next_h);
  THTensor_(cadd)(grad_prev_c, grad_next_c, 1, grad_output_gate);

  // we use above grad_input_gate to compute grad_output_gate
  THTensor_(fill)(grad_output_gate, 1);
  THTensor_(cadd)(grad_output_gate, grad_output_gate, -1, output_gate);
  THTensor_(cmul)(grad_output_gate, grad_output_gate, output_gate);
  THTensor_(cmul)(grad_output_gate, grad_output_gate, grad_input_gate);
  THTensor_(cmul)(grad_output_gate, grad_output_gate, grad_next_h);

  // Use grad_input_gate as a temporary buffer for computing grad_input_transform
  THTensor_(cmul)(grad_input_gate, input_transform, input_transform);
  THTensor_(fill)(grad_input_transform, 1);
  THTensor_(cadd)(grad_input_transform, grad_input_transform, -1, grad_input_gate);
  THTensor_(cmul)(grad_input_transform, grad_input_transform, input_gate);
  THTensor_(cmul)(grad_input_transform, grad_input_transform, grad_prev_c);

  // We don't need any temporary storage for these so do them last
  THTensor_(fill)(grad_input_gate, 1);
  THTensor_(cadd)(grad_input_gate, grad_input_gate, -1, input_gate);
  THTensor_(cmul)(grad_input_gate, grad_input_gate, input_gate);
  THTensor_(cmul)(grad_input_gate, grad_input_gate, input_transform);
  THTensor_(cmul)(grad_input_gate, grad_input_gate, grad_prev_c);

  THTensor_(fill)(grad_forget_gate, 1);
  THTensor_(cadd)(grad_forget_gate, grad_forget_gate, -1, forget_gate);
  THTensor_(cmul)(grad_forget_gate, grad_forget_gate, forget_gate);
  THTensor_(cmul)(grad_forget_gate, grad_forget_gate, prev_c);
  THTensor_(cmul)(grad_forget_gate, grad_forget_gate, grad_prev_c);

  // now for the main dish
  THTensor *Wx_t = THTensor_(newTranspose)(Wx, 0, 1);
  THTensor *Wh_t = THTensor_(newTranspose)(Wh, 0, 1);
  THTensor *cur_x_t = THTensor_(newTranspose)(cur_x, 0, 1);
  THTensor *prev_h_t = THTensor_(newTranspose)(prev_h, 0, 1);

  THTensor_(addmm)(grad_cur_x, 0, grad_cur_x, 1, grad_gates, Wx_t);
  THTensor_(addmm)(grad_Wx, 1, grad_Wx, scale, cur_x_t, grad_gates);
  THTensor_(addmm)(grad_Wh, 1, grad_Wh, scale, prev_h_t, grad_gates);
  THTensor_(resize2d)(grad_gates_sum, 1, 4 * hiddensize);
  THTensor_(sum)(grad_gates_sum, grad_gates, 0, 0);
  THTensor_(cadd)(grad_b, grad_b, scale, grad_gates_sum);

  THTensor_(addmm)(grad_prev_h, 0, grad_prev_h, 1, grad_gates, Wh_t);
  THTensor_(cmul)(grad_prev_c, grad_prev_c, forget_gate);

  THTensor_(free)(Wx);
  THTensor_(free)(Wh);
  THTensor_(free)(input_gate);
  THTensor_(free)(forget_gate);
  THTensor_(free)(output_gate);
  THTensor_(free)(input_transform);

  THTensor_(free)(grad_Wx);
  THTensor_(free)(grad_Wh);
  THTensor_(free)(grad_input_gate);
  THTensor_(free)(grad_forget_gate);
  THTensor_(free)(grad_output_gate);
  THTensor_(free)(grad_input_transform);

  THTensor_(free)(Wx_t);
  THTensor_(free)(Wh_t);
  THTensor_(free)(cur_x_t);
  THTensor_(free)(prev_h_t);

  return 3;
}

static const struct luaL_Reg nn_(StepLSTM__) [] = {
  {"StepLSTM_updateOutput", nn_(StepLSTM_updateOutput)},
  {"StepLSTM_backward", nn_(StepLSTM_backward)},
  {NULL, NULL}
};

static void nn_(StepLSTM_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(StepLSTM__), "nn");
  lua_pop(L,1);
}

#endif
