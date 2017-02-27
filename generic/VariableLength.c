#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VariableLength.c"
#else

static int nn_(from_samples_to_structured)(lua_State *L) {
  // processes inputs
  if (lua_gettop(L) != 3)
    return LUA_HANDLE_ERROR_STR(L, "expected 3 arguments: samples, output, mask");
  const int samples_index = 1;
  const int output_index = 2;
  const int mask_index = 3;
  if (!lua_istable(L, samples_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for first argument");
  THTensor *output = luaT_checkudata(L, output_index, torch_Tensor);
  if (!THTensor_(isContiguous)(output))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");
  THByteTensor *mask = luaT_checkudata(L, mask_index, "torch.ByteTensor");
  if (!THByteTensor_isContiguous(mask))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");

  // loads all samples from the table
  long n_samples = lua_objlen(L, samples_index);
  THTensor *tensors[n_samples];
  lua_pushnil(L);
  while (lua_next(L, samples_index) != 0) {
    long index = lua_tointeger(L, -2);
    THTensor *tensor = (THTensor *)luaT_checkudata(L, -1, torch_Tensor);
    tensors[index-1] = tensor;
    lua_pop(L, 1);
  }

  // processes the samples to get some meta-info that will be used to determine the positioning in
  // the dense tensor created in the output
  Sample samples_info[n_samples];
  THTensor* step = THTensor_(new)(); // a tensor that contains first step of first tensor
  THTensor* _step = THTensor_(new)(); // contains first step of other tensors (sizes much match)
  for (long i = 0; i < n_samples; i++) {
    THTensor_(narrow)(_step, tensors[i], 0, 0, 1); // 1 [x ...]
    if (i == 0)
      THTensor_(narrow)(step, tensors[i], 0, 0, 1);
    else if (!THTensor_(isSameSizeAs)(step, _step))
      return LUA_HANDLE_ERROR_STR(L, "got tensors of different sizes");
    samples_info[i].length = THTensor_(size)(tensors[i], 0);
    samples_info[i].index = i;
    samples_info[i].assigned_row = -1;
  }

  // sorts samples in order of length
  qsort(samples_info, n_samples, sizeof(Sample), sample_compare);

  long max_length = samples_info[n_samples-1].length;

  // creates the two tables with meta-info that will be output
  lua_newtable(L);
  const int indexes_index = lua_gettop(L);
  int local_indexes_index = 0;
  lua_newtable(L);
  const int mapped_lengths_index = lua_gettop(L);
  int local_mapped_lengths_index = 0;

  long row_index = 0;
  long length_available = max_length;
  long count = 0, row_count = 0;
  long start_index = 0;

  // while there are unprocessed samples...
  while (count < n_samples) {
    // flag of whether a sample was added in this iteration
    int added_sample = 0;
    // for each sample provided
    for (long i = n_samples-1; i >= 0; i--) {
      // checks if the current sample hasn't been assigned yet and fits the space left in the line
      if (samples_info[i].assigned_row == -1 && samples_info[i].length <= length_available) {
        long sample_index = samples_info[i].index;

        // if first sample in the row, creates sub-tables with meta-info for each row
        if (row_count == 0) {
          lua_newtable(L);
          local_indexes_index = lua_gettop(L);
          lua_newtable(L);
          local_mapped_lengths_index = lua_gettop(L);
        }

        // places the meta-info about the sample (index and length) into the tables
        row_count++;
        lua_pushinteger(L, sample_index+1);
        lua_rawseti(L, local_indexes_index, row_count);
        lua_pushinteger(L, samples_info[i].length);
        lua_rawseti(L, local_mapped_lengths_index, row_count);

        // assigns the sample to this row and updates the row and sample info
        samples_info[i].assigned_row = row_index;
        length_available -= samples_info[i].length + 1;
        start_index += samples_info[i].length + 1;
        count++;
        added_sample = 1;
      }
    }

    // if no sample was added, it means no sample available can fit in the space left, so we have to
    // add another table
    if (!added_sample) {
      // saves the current row-based meta-info
      lua_rawseti(L, mapped_lengths_index, row_index+1);
      lua_rawseti(L, indexes_index, row_index+1);
      // and advances rows
      row_index++;
      length_available = max_length;
      start_index = 0;
      row_count = 0;
    }
  }
  // saves the last row's meta-info
  lua_rawseti(L, mapped_lengths_index, row_index+1);
  lua_rawseti(L, indexes_index, row_index+1);

  // with the info available, resizes the output and mask
  long n_rows = lua_objlen(L, indexes_index);
  // output will have size: maxlen x nrows [x ...]
  long output_dim = THTensor_(nDimension)(step) + 1;
  THLongStorage* output_size = THLongStorage_newWithSize(output_dim);
  output_size->data[0] = max_length;
  output_size->data[1] = n_rows;
  for (long i=2; i < output_dim; i++) {
    output_size->data[i] = THTensor_(size)(step, i-1);
  }
  THTensor_(resize)(output, output_size, NULL);
  THByteTensor_resize2d(mask, max_length, n_rows);
  // mask starts filled with ones indicating it's empty
  THByteTensor_fill(mask, 1);

  THTensor *row = THTensor_(new)(), *section = THTensor_(new)();
  THByteTensor *mrow = THByteTensor_new(), *msection = THByteTensor_new();
  // for each row in the output
  for (long i = 0; i < n_rows; i++) {
    THTensor_(select)(row, output, 1, i);
    THByteTensor_select(mrow, mask, 1, i);
    lua_rawgeti(L, indexes_index, i+1);
    const int local_indexes_index = lua_gettop(L);
    lua_rawgeti(L, mapped_lengths_index, i+1);
    const int local_mapped_lengths_index = lua_gettop(L);

    long n_entries_in_row = lua_objlen(L, -1);
    long start = 0;
    // for each sample placed in that row
    for (long j = 0; j < n_entries_in_row; j++) {
      lua_rawgeti(L, local_indexes_index, j+1);
      lua_rawgeti(L, local_mapped_lengths_index, j+1);
      long index = lua_tointeger(L, -2);
      long length = lua_tointeger(L, -1);
      lua_pop(L, 2);

      // copies the data from the input and fills the mask
      THTensor_(narrow)(section, row, 0, start, length);
      THByteTensor_narrow(msection, mrow, 0, start, length);
      THTensor_(copy)(section, tensors[index-1]);
      THByteTensor_fill(msection, 0);
      start += length + 1;
    }
    lua_pop(L, 2);
  }
  THTensor_(free)(row);
  THTensor_(free)(section);
  THTensor_(free)(step);
  THTensor_(free)(_step);
  THLongStorage_free(output_size);
  THByteTensor_free(mrow);
  THByteTensor_free(msection);

  return 2;
}

// converts the dense tensor `input` into a list of samples `output`, each with its correct length.
static int nn_(from_structured_to_samples)(lua_State *L) {
  // processes inputs
  if (lua_gettop(L) != 3)
    return LUA_HANDLE_ERROR_STR(L, "expected 3 arguments: indexing, lengths, input");
  const int indexes_index = 1;
  const int mapped_lengths_index = 2;
  const int input_index = 3;
  if (!lua_istable(L, indexes_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for first argument");
  if (!lua_istable(L, mapped_lengths_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for second argument");

  THTensor *input = luaT_checkudata(L, input_index, torch_Tensor);
  if (!THTensor_(isContiguous)(input))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");

  lua_newtable(L);
  const int output_index = lua_gettop(L);

  long n_rows = lua_objlen(L, indexes_index);
  THTensor *row = THTensor_(new)();
  // for each row in the input
  for (long i = 0; i < n_rows; i++) {
    THTensor_(select)(row, input, 1, i);
    lua_rawgeti(L, indexes_index, i+1);
    const int local_indexes_index = lua_gettop(L);
    lua_rawgeti(L, mapped_lengths_index, i+1);
    const int local_mapped_lengths_index = lua_gettop(L);

    long n_entries_in_row = lua_objlen(L, -1);
    long start = 0;
    // for each sample placed in that row
    for (long j = 0; j < n_entries_in_row; j++) {
      lua_rawgeti(L, local_indexes_index, j+1);
      lua_rawgeti(L, local_mapped_lengths_index, j+1);
      long index = lua_tointeger(L, -2);
      long length = lua_tointeger(L, -1);
      lua_pop(L, 2);

      // gets the sub-tensor of the row that corresponds to the sample and places in the table
      THTensor *dest = THTensor_(new)();
      THTensor_(narrow)(dest, row, 0, start, length);
      start += length + 1;
      luaT_pushudata(L, dest, torch_Tensor);
      lua_rawseti(L, output_index, index);
    }
    lua_pop(L, 2);
  }
  THTensor_(free)(row);

  return 1;
}

static int nn_(from_structured_to_final)(lua_State *L) {
  // processes inputs
  if (lua_gettop(L) != 4)
    return LUA_HANDLE_ERROR_STR(L, "expected 4 arguments: indexing, lengths, input, output");
  const int indexes_index = 1;
  const int mapped_lengths_index = 2;
  const int input_index = 3;
  const int output_index = 4;
  if (!lua_istable(L, indexes_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for first argument");
  if (!lua_istable(L, mapped_lengths_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for second argument");

  THTensor *input = luaT_checkudata(L, input_index, torch_Tensor);
  if (!THTensor_(isContiguous)(input))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");
  THTensor *output = luaT_checkudata(L, output_index, torch_Tensor);
  if (!THTensor_(isContiguous)(output))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");

  long n_samples = get_n_samples(L, mapped_lengths_index);
  long output_dim = THTensor_(nDimension)(input) - 1;
  THLongStorage* output_size = THLongStorage_newWithSize(output_dim); // n_samples [x ...]
  output_size->data[0] = n_samples;
  for (long i=1;i < output_dim; i++){
    output_size->data[i] = THTensor_(size)(input, i+1);
  }
  THTensor_(resize)(output, output_size, NULL);

  long n_rows = lua_objlen(L, indexes_index);
  THTensor *row = THTensor_(new)(), *section = THTensor_(new)();
  THTensor *output_section = THTensor_(new)();
  // for each row in the output
  for (long i = 0; i < n_rows; i++) {
    THTensor_(select)(row, input, 1, i);
    lua_rawgeti(L, indexes_index, i+1);
    const int local_indexes_index = lua_gettop(L);
    lua_rawgeti(L, mapped_lengths_index, i+1);
    const int local_mapped_lengths_index = lua_gettop(L);

    long n_entries_in_row = lua_objlen(L, -1);
    long start = 0;
    // for each sample placed in that row
    for (long j = 0; j < n_entries_in_row; j++) {
      lua_rawgeti(L, local_indexes_index, j+1);
      lua_rawgeti(L, local_mapped_lengths_index, j+1);
      long index = lua_tointeger(L, -2);
      long length = lua_tointeger(L, -1);
      lua_pop(L, 2);

      // gets the sub-tensor of the row that corresponds to the sample and places in the table
      THTensor_(select)(section, row, 0, start + length-1);
      THTensor_(select)(output_section, output, 0, index-1);
      THTensor_(copy)(output_section, section);
      start += length + 1;
    }
    lua_pop(L, 2);
  }
  THTensor_(free)(row);
  THTensor_(free)(section);
  THTensor_(free)(output_section);
  THLongStorage_free(output_size);

  return 0;
}

static int nn_(from_final_to_structured)(lua_State *L) {
  if (lua_gettop(L) != 4)
    return LUA_HANDLE_ERROR_STR(L, "expected 4 arguments: indexing, lengths, input, output");
  const int indexes_index = 1;
  const int mapped_lengths_index = 2;
  const int input_index = 3;
  const int output_index = 4;
  if (!lua_istable(L, indexes_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for first argument");
  if (!lua_istable(L, mapped_lengths_index))
    return LUA_HANDLE_ERROR_STR(L, "expected table for second argument");

  THTensor *input = luaT_checkudata(L, input_index, torch_Tensor);
  if (!THTensor_(isContiguous)(input))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");
  THTensor *output = luaT_checkudata(L, output_index, torch_Tensor);
  if (!THTensor_(isContiguous)(output))
    return LUA_HANDLE_ERROR_STR(L, "tensor should be contiguous");

  long max_length = get_max_length(L, mapped_lengths_index);
  long n_rows = lua_objlen(L, mapped_lengths_index);

  long output_dim = THTensor_(nDimension)(input) + 1;
  THLongStorage* output_size = THLongStorage_newWithSize(output_dim); // max_length x n_rows [x ...]
  output_size->data[0] = max_length;
  output_size->data[1] = n_rows;
  for (long i=2;i < output_dim; i++){
    output_size->data[i] = THTensor_(size)(input, i-1);
  }
  THTensor_(resize)(output, output_size, NULL);
  THTensor_(fill)(output, 0);

  THTensor *row = THTensor_(new)(), *section = THTensor_(new)();
  THTensor *input_section = THTensor_(new)();
  // for each row in the input
  for (long i = 0; i < n_rows; i++) {
    THTensor_(select)(row, output, 1, i);
    lua_rawgeti(L, indexes_index, i+1);
    const int local_indexes_index = lua_gettop(L);
    lua_rawgeti(L, mapped_lengths_index, i+1);
    const int local_mapped_lengths_index = lua_gettop(L);

    long n_entries_in_row = lua_objlen(L, -1);
    long start = 0;
    // for each sample placed in that row
    for (long j = 0; j < n_entries_in_row; j++) {
      lua_rawgeti(L, local_indexes_index, j+1);
      lua_rawgeti(L, local_mapped_lengths_index, j+1);
      long index = lua_tointeger(L, -2);
      long length = lua_tointeger(L, -1);
      lua_pop(L, 2);

      // copies the data from the input
      THTensor_(select)(section, row, 0, start + length-1);
      THTensor_(select)(input_section, input, 0, index-1);
      THTensor_(copy)(section, input_section);
      start += length + 1;
    }
    lua_pop(L, 2);
  }
  THTensor_(free)(row);
  THTensor_(free)(section);
  THTensor_(free)(input_section);
  THLongStorage_free(output_size);

  return 0;
}

static const struct luaL_Reg nn_(VariableLength__) [] = {
  {"VariableLength_FromSamples", nn_(from_samples_to_structured)},
  {"VariableLength_ToSamples", nn_(from_structured_to_samples)},
  {"VariableLength_ToFinal", nn_(from_structured_to_final)},
  {"VariableLength_FromFinal", nn_(from_final_to_structured)},
  {NULL, NULL}
};

static void nn_(VariableLength_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VariableLength__), "nn");
  lua_pop(L,1);
}

#endif
