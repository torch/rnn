#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StepLSTM.c"
#else

static int nn_(StepLSTM_updateOutput)(lua_State *L) {
  THTensor *cur_x = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *prev_c = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *next_h = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *next_c = luaT_checkudata(L, 6, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  int inputsize = luaT_getfieldcheckint(L, 1, "inputsize");
  int outputsize = luaT_getfieldcheckint(L, 1, "outputsize");
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *buffer = luaT_getfieldcheckudata(L, 1, "buffer", torch_Tensor);
  THTensor *Wx = luaT_getfieldcheckudata(L, 1, "Wx", torch_Tensor);
  THTensor *Wh = luaT_getfieldcheckudata(L, 1, "Wh", torch_Tensor);

  THTensor_(set)(buffer, bias);
  THTensor_(resize2d)(buffer, 1, 4 * outputsize);
  buffer->stride[0] = 0;
  buffer->size[0] = batchsize;

  THTensor_(narrow)(Wx, weight, 0, 0, inputsize);
  THTensor_(narrow)(Wh, weight, 0, inputsize, outputsize);

  THTensor_(resize2d)(next_h, batchsize, outputsize);
  THTensor_(resize2d)(next_c, batchsize, outputsize);

  THTensor *gates = luaT_getfieldcheckudata(L, 1, "gates", torch_Tensor);
  THTensor_(resize2d)(gates, batchsize, 4 * outputsize);
  THTensor_(fill)(gates, 0);

  // forward
  THTensor_(addmm)(gates, 1, buffer, 1, cur_x, Wx);
  THTensor_(addmm)(gates, 1, gates, 1, prev_h, Wh);

  THTensor_(narrow)(buffer, gates, 1, 0, 3 * outputsize);
  THTensor_(sigmoid)(buffer, buffer);

  THTensor_(narrow)(buffer, gates, 1, 3 * outputsize, outputsize);
  THTensor_(tanh)(buffer, buffer);

  THTensor *input_gate = luaT_getfieldcheckudata(L, 1, "input_gate", torch_Tensor);
  THTensor *forget_gate = luaT_getfieldcheckudata(L, 1, "forget_gate", torch_Tensor);
  THTensor *output_gate = luaT_getfieldcheckudata(L, 1, "output_gate", torch_Tensor);
  THTensor *input_transform = luaT_getfieldcheckudata(L, 1, "input_transform", torch_Tensor);

  THTensor_(narrow)(input_gate, gates, 1, 0, outputsize);
  THTensor_(narrow)(forget_gate, gates, 1, outputsize, outputsize);
  THTensor_(narrow)(output_gate, gates, 1, 2*outputsize, outputsize);
  THTensor_(narrow)(input_transform, gates, 1, 3*outputsize, outputsize);

  THTensor_(cmul)(next_h, input_gate, input_transform);
  THTensor_(cmul)(next_c, forget_gate, prev_c);
  THTensor_(cadd)(next_c, next_c, 1, next_h);
  THTensor_(tanh)(next_h, next_c);
  THTensor_(cmul)(next_h, next_h, output_gate);

  return 2;
}

static int nn_(StepLSTM_backward)(lua_State *L) {
  THTensor *cur_x = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *prev_h = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *prev_c = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *next_c = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *grad_next_h = luaT_checkudata(L, 6, torch_Tensor);
  THTensor *grad_next_c = luaT_checkudata(L, 7, torch_Tensor);
  lua_Number scale = luaL_checknumber(L, 8);
  THTensor *grad_gates = luaT_checkudata(L, 9, torch_Tensor);
  THTensor *grad_gates_sum = luaT_checkudata(L, 10, torch_Tensor);
  THTensor *grad_cur_x = luaT_checkudata(L, 11, torch_Tensor);
  THTensor *grad_prev_h = luaT_checkudata(L, 12, torch_Tensor);
  THTensor *grad_prev_c = luaT_checkudata(L, 13, torch_Tensor);

  int batchsize = THTensor_(size)(cur_x, 0);
  int inputsize = luaT_getfieldcheckint(L, 1, "inputsize");
  int outputsize = luaT_getfieldcheckint(L, 1, "outputsize");
  if (THTensor_(size)(cur_x, 1) != inputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected input[1]:size(2) == inputsize");
  if (THTensor_(size)(grad_next_h, 1) != outputsize)
    return LUA_HANDLE_ERROR_STR(L, "expected gradOutput[1]:size(2) == outputsize");

  THTensor_(resize2d)(grad_cur_x, batchsize, inputsize);
  THTensor_(resize2d)(grad_prev_h, batchsize, outputsize);
  THTensor_(resize2d)(grad_prev_c, batchsize, outputsize);

  // these tensors were set-up in updateOutput
  THTensor *Wx = luaT_getfieldcheckudata(L, 1, "Wx", torch_Tensor);
  THTensor *Wh = luaT_getfieldcheckudata(L, 1, "Wh", torch_Tensor);
  THTensor *input_gate = luaT_getfieldcheckudata(L, 1, "input_gate", torch_Tensor);
  THTensor *forget_gate = luaT_getfieldcheckudata(L, 1, "forget_gate", torch_Tensor);
  THTensor *output_gate = luaT_getfieldcheckudata(L, 1, "output_gate", torch_Tensor);
  THTensor *input_transform = luaT_getfieldcheckudata(L, 1, "input_transform", torch_Tensor);

  // set-up grad tensors
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *grad_b = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor *grad_Wx = luaT_getfieldcheckudata(L, 1, "grad_Wx", torch_Tensor);
  THTensor *grad_Wh = luaT_getfieldcheckudata(L, 1, "grad_Wh", torch_Tensor);
  THTensor_(narrow)(grad_Wx, gradWeight, 0, 0, inputsize);
  THTensor_(narrow)(grad_Wh, gradWeight, 0, inputsize, outputsize);

  THTensor_(resize2d)(grad_gates, batchsize, 4 * outputsize);
  THTensor_(fill)(grad_gates, 0);

  THTensor *grad_input_gate = luaT_getfieldcheckudata(L, 1, "grad_input_gate", torch_Tensor);
  THTensor *grad_forget_gate = luaT_getfieldcheckudata(L, 1, "grad_forget_gate", torch_Tensor);
  THTensor *grad_output_gate = luaT_getfieldcheckudata(L, 1, "grad_output_gate", torch_Tensor);
  THTensor *grad_input_transform = luaT_getfieldcheckudata(L, 1, "grad_input_transform", torch_Tensor);

  THTensor_(narrow)(grad_input_gate, grad_gates, 1, 0, outputsize);
  THTensor_(narrow)(grad_forget_gate, grad_gates, 1, outputsize, outputsize);
  THTensor_(narrow)(grad_output_gate, grad_gates, 1, 2*outputsize, outputsize);
  THTensor_(narrow)(grad_input_transform, grad_gates, 1, 3*outputsize, outputsize);

  // backward

  // we use grad_[input,forget,output]_gate as temporary buffers to compute grad_prev_c.
  THTensor_(tanh)(grad_input_gate, next_c);
  THTensor_(cmul)(grad_forget_gate, grad_input_gate, grad_input_gate);
  //grad_output_gate:fill(1):add(-1, grad_forget_gate):cmul(output_gate):cmul(grad_next_h)
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
  THTensor_(resize2d)(grad_gates_sum, 1, 4 * outputsize);
  THTensor_(sum)(grad_gates_sum, grad_gates, 0);
  THTensor_(cadd)(grad_b, grad_b, scale, grad_gates_sum);

  THTensor_(addmm)(grad_prev_h, 0, grad_prev_h, 1, grad_gates, Wh_t);
  THTensor_(cmul)(grad_prev_c, grad_prev_c, forget_gate);

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
