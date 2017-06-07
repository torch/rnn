#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda/StepLSTM.cu"
#else

#if defined(THC_REAL_IS_HALF)
#define _REAL(val) THC_float2half(val)
#else
#define _REAL(val) (val)
#endif

static int nn_(StepLSTM_updateOutput)(lua_State *L) {
  THCState *state = getCudaState(L);
  THCTensor *weight = (THCTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *bias = (THCTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THCTensor *gates = (THCTensor *)luaT_checkudata(L, 3, torch_Tensor);
  THCTensor *cur_x = (THCTensor *)luaT_checkudata(L, 4, torch_Tensor);
  THCTensor *prev_h = (THCTensor *)luaT_checkudata(L, 5, torch_Tensor);
  THCTensor *prev_c = (THCTensor *)luaT_checkudata(L, 6, torch_Tensor);
  int inputsize = luaL_checkinteger(L, 7);
  int hiddensize = luaL_checkinteger(L, 8);
  int outputsize = luaL_checkinteger(L, 9);
  THCTensor *next_h = (THCTensor *)luaT_checkudata(L, 10, torch_Tensor); // when LSTMP pass hidden[t]
  THCTensor *next_c = (THCTensor *)luaT_checkudata(L, 11, torch_Tensor);

  int batchsize = THCTensor_(size)(state, cur_x, 0);
  if (THCTensor_(size)(state, cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");

  THLongStorage* size = THLongStorage_newWithSize2(1, 4 * hiddensize);
  THCTensor *buffer = THCTensor_(newView)(state, bias, size);
  buffer->stride[0] = 0;
  buffer->size[0] = batchsize;

  THCTensor *Wx = THCTensor_(newNarrow)(state, weight, 0, 0, inputsize);
  THCTensor *Wh = THCTensor_(newNarrow)(state, weight, 0, inputsize, outputsize);

  THCTensor_(resize2d)(state, next_h, batchsize, hiddensize);
  THCTensor_(resize2d)(state, next_c, batchsize, hiddensize);
  long nElement = THCTensor_(nElement)(state, gates);
  THCTensor_(resize2d)(state, gates, batchsize, 4 * hiddensize);
  if (nElement != batchsize * 4 * hiddensize)
    THCTensor_(fill)(state, gates, _REAL(0));

  // forward
  THCTensor_(addmm)(state, gates, _REAL(1), buffer, _REAL(1), cur_x, Wx);
  THCTensor_(addmm)(state, gates, _REAL(1), gates, _REAL(1), prev_h, Wh);

  THCTensor_(narrow)(state, buffer, gates, 1, 0, 3 * hiddensize);
  THCTensor_(sigmoid)(state, buffer, buffer);

  THCTensor_(narrow)(state, buffer, gates, 1, 3 * hiddensize, hiddensize);
  THCTensor_(tanh)(state, buffer, buffer);

  THCTensor *input_gate = THCTensor_(newNarrow)(state, gates, 1, 0, hiddensize);
  THCTensor *forget_gate = THCTensor_(newNarrow)(state, gates, 1, hiddensize, hiddensize);
  THCTensor *output_gate = THCTensor_(newNarrow)(state, gates, 1, 2*hiddensize, hiddensize);
  THCTensor *input_transform = THCTensor_(newNarrow)(state, gates, 1, 3*hiddensize, hiddensize);

  THCTensor_(cmul)(state, next_h, input_gate, input_transform);
  THCTensor_(cmul)(state, next_c, forget_gate, prev_c);
  THCTensor_(cadd)(state, next_c, next_c, _REAL(1), next_h);
  THCTensor_(tanh)(state, next_h, next_c);
  THCTensor_(cmul)(state, next_h, next_h, output_gate);

  THCTensor_(free)(state, Wx);
  THCTensor_(free)(state, Wh);
  THCTensor_(free)(state, buffer);
  THCTensor_(free)(state, input_gate);
  THCTensor_(free)(state, forget_gate);
  THCTensor_(free)(state, output_gate);
  THCTensor_(free)(state, input_transform);
  THLongStorage_free(size);

  if (lua_gettop(L) > 11) // implements LSTMP (P stands for projection layer)
  {
    THCTensor *hidden = next_h;
    THCTensor *weightO = (THCTensor *)luaT_checkudata(L, 12, torch_Tensor);
    next_h = (THCTensor *)luaT_checkudata(L, 13, torch_Tensor);
    THCTensor_(resize2d)(state, next_h, batchsize, outputsize);
    THCTensor_(addmm)(state, next_h, _REAL(0), next_h, _REAL(1), hidden, weightO);
    // push results onto stack
    luaT_pushudata(L, next_c, torch_Tensor);
  }

  return 2;
}

static int nn_(StepLSTM_backward)(lua_State *L) {
  THCState *state = getCudaState(L);
  THCTensor *weight = (THCTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *gates = (THCTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THCTensor *gradWeight = (THCTensor *)luaT_checkudata(L, 3, torch_Tensor);
  THCTensor *grad_b = (THCTensor *)luaT_checkudata(L, 4, torch_Tensor);
  THCTensor *grad_gates = (THCTensor *)luaT_checkudata(L, 5, torch_Tensor);
  THCTensor *grad_gates_sum = (THCTensor *)luaT_checkudata(L, 6, torch_Tensor);
  THCTensor *cur_x = (THCTensor *)luaT_checkudata(L, 7, torch_Tensor);
  THCTensor *prev_h = (THCTensor *)luaT_checkudata(L, 8, torch_Tensor);
  THCTensor *prev_c = (THCTensor *)luaT_checkudata(L, 9, torch_Tensor);
  THCTensor *next_c = (THCTensor *)luaT_checkudata(L, 10, torch_Tensor);
  THCTensor *grad_next_h = (THCTensor *)luaT_checkudata(L, 11, torch_Tensor);
  THCTensor *grad_next_c = (THCTensor *)luaT_checkudata(L, 12, torch_Tensor);
  lua_Number scale = luaL_checknumber(L, 13);
  int inputsize = luaL_checkinteger(L, 14);
  int hiddensize = luaL_checkinteger(L, 15);
  int outputsize = luaL_checkinteger(L, 16);
  THCTensor *grad_cur_x = (THCTensor *)luaT_checkudata(L, 17, torch_Tensor);
  THCTensor *grad_prev_h = (THCTensor *)luaT_checkudata(L, 18, torch_Tensor);
  THCTensor *grad_prev_c = (THCTensor *)luaT_checkudata(L, 19, torch_Tensor);

  int batchsize = THCTensor_(size)(state, cur_x, 0);
  if (THCTensor_(size)(state, cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");
  if (THCTensor_(size)(state, grad_next_h, 1) != outputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected gradOutput[1]:size(2) == outputsize");

  if (lua_gettop(L) > 19) // LSTMP
  {
    THCTensor *weightO = (THCTensor *)luaT_checkudata(L, 20, torch_Tensor);
    THCTensor *hidden = (THCTensor *)luaT_checkudata(L, 21, torch_Tensor);
    THCTensor *gradWeightO = (THCTensor *)luaT_checkudata(L, 22, torch_Tensor);
    THCTensor *grad_hidden = (THCTensor *)luaT_checkudata(L, 23, torch_Tensor);

    THCTensor *hidden_t = THCTensor_(newTranspose)(state, hidden, 0, 1);
    THCTensor *weightO_t = THCTensor_(newTranspose)(state, weightO, 0, 1);

    THCTensor_(addmm)(state, gradWeightO, _REAL(scale), gradWeightO, _REAL(1), hidden_t, grad_next_h);
    THCTensor_(resize2d)(state, grad_hidden, batchsize, hiddensize);
    THCTensor_(addmm)(state, grad_hidden, _REAL(0), grad_hidden, _REAL(1), grad_next_h, weightO_t);

    grad_next_h = grad_hidden;

    THCTensor_(free)(state, hidden_t);
    THCTensor_(free)(state, weightO_t);

    // push results to top of stack
    luaT_pushudata(L, grad_cur_x, torch_Tensor);
    luaT_pushudata(L, grad_prev_h, torch_Tensor);
    luaT_pushudata(L, grad_prev_c, torch_Tensor);
  }

  THCTensor_(resize2d)(state, grad_cur_x, batchsize, inputsize);
  THCTensor_(resize2d)(state, grad_prev_h, batchsize, outputsize);
  THCTensor_(resize2d)(state, grad_prev_c, batchsize, hiddensize);

  // these tensors were set-up in updateOutput
  THCTensor *Wx = THCTensor_(newNarrow)(state, weight, 0, 0, inputsize);
  THCTensor *Wh = THCTensor_(newNarrow)(state, weight, 0, inputsize, outputsize);

  THCTensor *input_gate = THCTensor_(newNarrow)(state, gates, 1, 0, hiddensize);
  THCTensor *forget_gate = THCTensor_(newNarrow)(state, gates, 1, hiddensize, hiddensize);
  THCTensor *output_gate = THCTensor_(newNarrow)(state, gates, 1, 2*hiddensize, hiddensize);
  THCTensor *input_transform = THCTensor_(newNarrow)(state, gates, 1, 3*hiddensize, hiddensize);

  // set-up grad tensors
  THCTensor *grad_Wx = THCTensor_(newNarrow)(state, gradWeight, 0, 0, inputsize);
  THCTensor *grad_Wh = THCTensor_(newNarrow)(state, gradWeight, 0, inputsize, outputsize);

  THCTensor_(resize2d)(state, grad_gates, batchsize, 4 * hiddensize);

  THCTensor *grad_input_gate = THCTensor_(newNarrow)(state, grad_gates, 1, 0, hiddensize);
  THCTensor *grad_forget_gate = THCTensor_(newNarrow)(state, grad_gates, 1, hiddensize, hiddensize);
  THCTensor *grad_output_gate = THCTensor_(newNarrow)(state, grad_gates, 1, 2*hiddensize, hiddensize);
  THCTensor *grad_input_transform = THCTensor_(newNarrow)(state, grad_gates, 1, 3*hiddensize, hiddensize);

  // backward

  // we use grad_[input,forget,output]_gate as temporary buffers to compute grad_prev_c.
  THCTensor_(tanh)(state, grad_input_gate, next_c);
  THCTensor_(cmul)(state, grad_forget_gate, grad_input_gate, grad_input_gate);

  THCTensor_(fill)(state, grad_output_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_output_gate, grad_output_gate, _REAL(-1), grad_forget_gate);
  THCTensor_(cmul)(state, grad_output_gate, grad_output_gate, output_gate);
  THCTensor_(cmul)(state, grad_output_gate, grad_output_gate, grad_next_h);
  THCTensor_(cadd)(state, grad_prev_c, grad_next_c, _REAL(1), grad_output_gate);

  // we use above grad_input_gate to compute grad_output_gate
  THCTensor_(fill)(state, grad_output_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_output_gate, grad_output_gate, _REAL(-1), output_gate);
  THCTensor_(cmul)(state, grad_output_gate, grad_output_gate, output_gate);
  THCTensor_(cmul)(state, grad_output_gate, grad_output_gate, grad_input_gate);
  THCTensor_(cmul)(state, grad_output_gate, grad_output_gate, grad_next_h);

  // Use grad_input_gate as a temporary buffer for computing grad_input_transform
  THCTensor_(cmul)(state, grad_input_gate, input_transform, input_transform);
  THCTensor_(fill)(state, grad_input_transform, _REAL(1));
  THCTensor_(cadd)(state, grad_input_transform, grad_input_transform, _REAL(-1), grad_input_gate);
  THCTensor_(cmul)(state, grad_input_transform, grad_input_transform, input_gate);
  THCTensor_(cmul)(state, grad_input_transform, grad_input_transform, grad_prev_c);

  // We don't need any temporary storage for these so do them last
  THCTensor_(fill)(state, grad_input_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_input_gate, grad_input_gate, _REAL(-1), input_gate);
  THCTensor_(cmul)(state, grad_input_gate, grad_input_gate, input_gate);
  THCTensor_(cmul)(state, grad_input_gate, grad_input_gate, input_transform);
  THCTensor_(cmul)(state, grad_input_gate, grad_input_gate, grad_prev_c);

  THCTensor_(fill)(state, grad_forget_gate, _REAL(1));
  THCTensor_(cadd)(state, grad_forget_gate, grad_forget_gate, _REAL(-1), forget_gate);
  THCTensor_(cmul)(state, grad_forget_gate, grad_forget_gate, forget_gate);
  THCTensor_(cmul)(state, grad_forget_gate, grad_forget_gate, prev_c);
  THCTensor_(cmul)(state, grad_forget_gate, grad_forget_gate, grad_prev_c);

  // now for the main dish
  THCTensor *Wx_t = THCTensor_(newTranspose)(state, Wx, 0, 1);
  THCTensor *Wh_t = THCTensor_(newTranspose)(state, Wh, 0, 1);
  THCTensor *cur_x_t = THCTensor_(newTranspose)(state, cur_x, 0, 1);
  THCTensor *prev_h_t = THCTensor_(newTranspose)(state, prev_h, 0, 1);

  THCTensor_(addmm)(state, grad_cur_x, _REAL(0), grad_cur_x, _REAL(1), grad_gates, Wx_t);
  THCTensor_(addmm)(state, grad_Wx, _REAL(1), grad_Wx, _REAL(scale), cur_x_t, grad_gates);
  THCTensor_(addmm)(state, grad_Wh, _REAL(1), grad_Wh, _REAL(scale), prev_h_t, grad_gates);
  THCTensor_(resize2d)(state, grad_gates_sum, 1, 4 * hiddensize);
  THCTensor_(sum)(state, grad_gates_sum, grad_gates, 0, 0);
  THCTensor_(cadd)(state, grad_b, grad_b, _REAL(scale), grad_gates_sum);

  THCTensor_(addmm)(state, grad_prev_h, _REAL(0), grad_prev_h, _REAL(1), grad_gates, Wh_t);
  THCTensor_(cmul)(state, grad_prev_c, grad_prev_c, forget_gate);

  THCTensor_(free)(state, Wx);
  THCTensor_(free)(state, Wh);
  THCTensor_(free)(state, input_gate);
  THCTensor_(free)(state, forget_gate);
  THCTensor_(free)(state, output_gate);
  THCTensor_(free)(state, input_transform);

  THCTensor_(free)(state, grad_Wx);
  THCTensor_(free)(state, grad_Wh);
  THCTensor_(free)(state, grad_input_gate);
  THCTensor_(free)(state, grad_forget_gate);
  THCTensor_(free)(state, grad_output_gate);
  THCTensor_(free)(state, grad_input_transform);

  THCTensor_(free)(state, Wx_t);
  THCTensor_(free)(state, Wh_t);
  THCTensor_(free)(state, cur_x_t);
  THCTensor_(free)(state, prev_h_t);

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

#undef _REAL
#endif
