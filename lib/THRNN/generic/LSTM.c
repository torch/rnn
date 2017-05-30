#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LSTM.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

/* Set tensor->size[0] MACRO */
#ifndef THRNN_LSTM_SET_SIZE
#define THRNN_LSTM_SET_SIZE(t, dim, newSize) ( t->size[dim] = newSize )
#endif

/* Set tensor->size[0] MACRO */
#ifndef THRNN_LSTM_SET_STRIDE
#define THRNN_LSTM_SET_STRIDE(t, dim, newStride) ( t->stride[dim] = newStride )
#endif

/* Increment storageOffset */
#ifndef THRNN_LSTM_INCREMENT_SOFFSET
#define THRNN_LSTM_INCREMENT_SOFFSET(t, dim, incrementSize) ( t->storageOffset += incrementSize*t->stride[dim] )
#endif

/* Decrement storageOffset */
#ifndef THRNN_LSTM_DECREMENT_SOFFSET
#define THRNN_LSTM_DECREMENT_SOFFSET(t, dim, decrementSize) ( t->storageOffset -= decrementSize*t->stride[dim] )
#endif

// structure to hold a bunch of tensors
// and delete them all together,
// increment their offset altogether,
// set their sizes/strides altogether, ...
struct THRNN_(buffer) {
   THTensor** array;
   THTensor* buf;
   int* sizes;
   int len;
   int total_size;
   int ndim;
   THLongStorage* default_buffer_sizes;
   /* declare as many members as desired, but the entire structure size must be known to the compiler. */
};

// create the buffer from an exisiting tensor (possibly NULL),
// and a LongStorage of default sizes for new individual buffers
struct THRNN_(buffer)* THRNN_(create_buffer)(THTensor* buf, THLongStorage* default_buffer_sizes)
{
   THTensor** arr;
   struct THRNN_(buffer)* x = malloc( sizeof( struct THRNN_(buffer) ));
   x->array = NULL;
   x->buf = buf? buf:THTensor_(new)();
   x->sizes = NULL;
   x->len = 0;
   x->total_size = 0;
   x->ndim = default_buffer_sizes->size;
   x->default_buffer_sizes = default_buffer_sizes;
   return x;
};

// This will add a tensor to the buffer, increasing the total size by
// 'size' times the product of sizes of all the other dimensions
THTensor* THRNN_(add_tensor_to_buffer)(struct THRNN_(buffer)* buffer, int size)
{
   buffer->len += 1;
   buffer->total_size += size;
   buffer->sizes = (int*)realloc(buffer->sizes, buffer->len * sizeof(int));
   buffer->sizes[buffer->len-1] = size;
   buffer->array = (THTensor**)realloc(buffer->array, buffer->len * sizeof(THTensor**));
   THTensor* new_guy = THTensor_(new)();
   buffer->array[buffer->len-1] = new_guy;
   return new_guy;
};

// Once the user has added all the tensors he needs to the buffer,
// he can 'compile' it, i.e. allocate one giant buffer that will be sliced
void THRNN_(compile_buffer)(struct THRNN_(buffer)* buffer)
{
   THLongStorage* bufferSize = THLongStorage_newWithSize(buffer->ndim);
   int lastDimSize = 0;
   int i;
   for (i = 0; i < buffer->len; i++)
   {
     lastDimSize += buffer->sizes[i];
   }
   for (i = 0; i < buffer->ndim-1; i++)
   {
     bufferSize->data[i] = buffer->default_buffer_sizes->data[i];
     THArgCheck(bufferSize->data[i] > 0, 1, "The buffer sizes must be > 0.");
   }
   bufferSize->data[buffer->ndim-1] = lastDimSize;

   THTensor_(resize)(buffer->buf, bufferSize, NULL);
   THLongStorage_free(bufferSize);
   int runningIdx = 0;
   for (i = 0; i < buffer->len; i++)
   {
     if (buffer->sizes[i])
     {
       THTensor_(narrow)(buffer->array[i], buffer->buf, buffer->ndim-1, runningIdx, buffer->sizes[i]);
     }
     runningIdx += buffer->sizes[i];
   }
   return;
};

// Delete the buffer and all its components
void THRNN_(delete_buffer)(struct THRNN_(buffer)* buffer, int delete_internal_buffer)
{
   int i;
   for (i = 0; i < buffer->len; i++)
   {
     THTensor_(free)(buffer->array[i]);
   }
   free(buffer->sizes);
   free(buffer->array);
   if (delete_internal_buffer)
     THTensor_(free)(buffer->buf);
   THLongStorage_free(buffer->default_buffer_sizes);
   free(buffer);
   return;
};

// batch set size
void THRNN_(buffer_set_size)(struct THRNN_(buffer)* buffer, int dim, int newSize)
{
  int i;
   for (i = 0; i < buffer->len; i++)
   {
     THRNN_LSTM_SET_SIZE(buffer->array[i], dim, newSize);
   }
}

// batch set stride
void THRNN_(buffer_set_stride)(struct THRNN_(buffer)* buffer, int dim, int newStride)
{
  int i;
   for (i = 0; i < buffer->len; i++)
   {
     THRNN_LSTM_SET_STRIDE(buffer->array[i], dim, newStride);
   }
}

// batch increment offset
void THRNN_(buffer_increment_soffset)(struct THRNN_(buffer)* buffer, int dim, int offset)
{
  int i;
   for (i = 0; i < buffer->len; i++)
   {
     THRNN_LSTM_INCREMENT_SOFFSET(buffer->array[i], dim, offset);
   }
}

// batch decrement offset
void THRNN_(buffer_decrement_soffset)(struct THRNN_(buffer)* buffer, int dim, int offset)
{
  int i;
   for (i = 0; i < buffer->len; i++)
   {
     THRNN_LSTM_DECREMENT_SOFFSET(buffer->array[i], dim, offset);
   }
}

// Convenient function to print tensors
static void THRNN_(printTensor)(char* name, THTensor *input)
{
  printf("Tensor %s\n", name);

  long ndim = THTensor_(nDimension)(input);
  int i;
  printf("- ndim %lu\n", ndim);
  for (i = 0; i < ndim; i++)
  {
    printf("\tdimension #%i:\n", i);
    int sz = THTensor_(size)(input, i);
    int st = THTensor_(stride)(input, i);
    printf("\t- size: %i\n", sz);
    printf("\t- stride: %i\n", st);
  }
  printf("\n");
}

// Convenient function to print the mean of a tensor
static void THRNN_(printMean)(char* name, THTensor *input)
{
  printf("Tensor %s\n", name);
  long nelements = THTensor_(nElement)(input);
  long ndim = THTensor_(nDimension)(input);
  accreal mean = THTensor_(meanall)(input);;
  int i;
  printf("\t- nelements %lu\n", nelements);
  printf("\t- mean: %g\n", mean);
}

#ifndef THRNN_LSTM_BASIC_LSTM_CELL
#define THRNN_LSTM_BASIC_LSTM_CELL 1
#endif

// Take a array of tensors, each tensor
// representing the inputs over time, or
// the last input in time if 'last' is == 1
// Here:
// input --> concatenation of these tensors over the first (batch) dimension
// sizes --> sizes of tensors
// This function computes:
// output --> transposed time-first of input,
//      i.e. for each time step, the elements are sequentially
//      aligned if they are from the first tensor, the second, ...
//      where first, second, ... are the sorted tensors by decreasing size
// sorted_by_batch_sizes --> the sizes sorted by batch size in decreasing order
// mapping --> the mapping from the output indices to the input indices. Useful for the inverse operation
void THRNN_(LSTM_bt_to_sorted_tb)(
          THRNNState    *state,
          THTensor      *input,
          THLongTensor  *sizes,
          THTensor      *output,
          THLongTensor  *mapping,
          THLongTensor  *sorted_by_batch_sizes,
          THLongTensor  *sorted_by_time_sizes,
          THLongTensor  *sorted_by_time_indices,
          int           last)
{
  long sizes_dim = THLongTensor_nDimension(sizes);
  long input_dim = THTensor_(nDimension)(input);
  THArgCheck(sizes_dim == 1, 2, "sizes must have 1 dimension");

  long input_size_0 = THTensor_(size)(input, 0);
  long nelements = THTensor_(nElement)(input);
  long nelements_left = nelements / input_size_0;
  long bsize = THLongTensor_size(sizes, 0);

  THLongTensor* sizes_copy = THLongTensor_new();
  THLongTensor* cumsum_sizes = THLongTensor_new();
  THLongTensor_sort(sorted_by_time_sizes, sorted_by_time_indices, sizes, 0, 1);
  THLongTensor_resize1d(sizes_copy, bsize);
  THLongTensor_cumsum(cumsum_sizes, sizes, 0);
  THLongTensor_copy(sizes_copy, sizes);


  long* sizes_data = THLongTensor_data(sizes);
  long* sizes_copy_data = THLongTensor_data(sizes_copy);
  long* cumsum_sizes_data = THLongTensor_data(cumsum_sizes);
  long* sorted_by_time_sizes_data = THLongTensor_data(sorted_by_time_sizes);
  long* sorted_by_time_indices_data = THLongTensor_data(sorted_by_time_indices);
  real* input_data = THTensor_(data)(input);

  THArgCheck(THLongTensor_isContiguous(sizes), 2, "sizes vector must be contiguous");

  long total_size = last?0:input_size_0;
  long i,b;
  if (last)
  {
    for (b = 0; b < bsize; b++)
    {
      total_size += sizes_data[b];
    }
  }

  // Resize the output
  long* new_size = malloc(input_dim*sizeof(long));
  long lnelements_left = nelements;

  for (i = 0;i < input_dim; i++)
  {
    new_size[i] = i?THTensor_(size)(input, i):total_size;
  }

  THTensor_(resizeNd)(output, input_dim, new_size, NULL);
  real* output_data = THTensor_(data)(output);
  free(new_size);

  // Make sure these inputs are contiguous to accelerate computations
  THArgCheck(THTensor_(isContiguous)(input), 1, "input vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output), 3, "output vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(mapping), 4, "mapping vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(sorted_by_batch_sizes), 5, "sorted_by_batch_sizes must be contiguous.");
  THArgCheck(THLongTensor_isContiguous(sorted_by_time_sizes), 6, "sorted_by_time_sizes must be contiguous.");
  THArgCheck(THLongTensor_isContiguous(sorted_by_time_indices), 7, "sorted_by_time_indices must be contiguous.");

  long idx = 0;
  THLongTensor_resize1d(mapping, total_size);
  THLongTensor_resize1d(sorted_by_batch_sizes, sorted_by_time_sizes_data[0]);
  THLongTensor_zero(sorted_by_batch_sizes);
  long* mapping_data = THLongTensor_data(mapping);
  long* sorted_by_batch_sizes_data = THLongTensor_data(sorted_by_batch_sizes);

  // if last, then the input only has end-of-sequence values defined
  // for each element of the batch
  if (last)
  {
    THTensor_(zero)(output);
//    THRNN_(printTensor)("output", output);
//    THRNN_(printTensor)("input", input);
    for (i = 0; i < sorted_by_time_sizes_data[0]; i++)
    {
      for (b = 0; b < bsize; b++)
      {
        long sidx = sorted_by_time_indices_data[b];
        long timesteps = sizes_copy_data[sidx];
        if (timesteps > 0)
        {
          long lidx = ((sidx>0?cumsum_sizes_data[sidx-1]:0) + sizes_data[sidx] - sizes_copy_data[sidx]);
          if (timesteps == 1)
          {
            memcpy(output_data + idx*nelements_left, input_data + sidx*nelements_left, nelements_left*sizeof(real));
          }
          mapping_data[lidx] = idx;
          sizes_copy_data[sidx]--;
          sorted_by_batch_sizes_data[i]++;
          idx++;
        }
      }
    }
  }
  else
  {
    for (i = 0; i < sorted_by_time_sizes_data[0]; i++)
    {
      for (b = 0; b < bsize; b++)
      {
        long sidx = sorted_by_time_indices_data[b];
        long timesteps = sizes_copy_data[sidx];
        if (timesteps > 0)
        {
          long lidx = ((sidx>0?cumsum_sizes_data[sidx-1]:0) + sizes_data[sidx] - sizes_copy_data[sidx]);
          memcpy(output_data + idx*nelements_left, input_data + lidx*nelements_left, nelements_left*sizeof(real));
          mapping_data[lidx] = idx;
          sizes_copy_data[sidx]--;
          sorted_by_batch_sizes_data[i]++;
          idx++;
        }
      }
    }
  }

  THLongTensor_free(sizes_copy);
  THLongTensor_free(cumsum_sizes);
}


// Reverse operation from LSTM_bt_to_sorted_tb
void THRNN_(LSTM_sorted_tb_to_bt)(
          THRNNState    *state,
          THTensor      *input,
          THLongTensor  *sizes,
          THLongTensor  *mapping,
          THTensor      *output,
          int           last)
{
  long input_dim = THTensor_(nDimension)(input);

  long input_size_0 = THTensor_(size)(input, 0);
  long nelements = THTensor_(nElement)(input);
  long nelements_left = nelements / input_size_0;
  long batch_size = THLongTensor_size(sizes, 0);

  // Resize the output
  long* new_size = malloc(input_dim*sizeof(long));
  long lnelements_left = nelements;
  long total_size = last?batch_size:input_size_0;
  long i,b;

  for (i = 0;i < input_dim; i++)
  {
    new_size[i] = i?THTensor_(size)(input, i):total_size;
  }

  THTensor_(resizeNd)(output, input_dim, new_size, NULL);
  real* output_data = THTensor_(data)(output);
  free(new_size);

  long* mapping_data = THLongTensor_data(mapping);
  real* input_data = THTensor_(data)(input);
  long* sizes_data = THLongTensor_data(sizes);

  // Make sure these inputs are contiguous to accelerate computations
  THArgCheck(THTensor_(isContiguous)(input), 1, "input vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(mapping), 3, "sizes vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output), 4, "output vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(sizes), 2, "sizes vector must be contiguous");


  if (last)
  {
    THLongTensor* cumsum_sizes = THLongTensor_new();
    THLongTensor_cumsum(cumsum_sizes, sizes, 0);
    long* cumsum_sizes_data = THLongTensor_data(cumsum_sizes);
    for (i = 0; i < batch_size; i++)
    {
      long idx = cumsum_sizes_data[i]-1;
      memcpy(output_data + i*nelements_left,
              input_data + mapping_data[idx]*nelements_left,
              nelements_left*sizeof(real));
    }
    THLongTensor_free(cumsum_sizes);
  }
  else
  {
    for (i = 0; i < input_size_0; i++)
    {
      memcpy(output_data + i*nelements_left,
              input_data + mapping_data[i]*nelements_left,
              nelements_left*sizeof(real));
    }
  }
}

/*
 * That macro is used in LSTM_updateOutput
 * and LSTM_backward, hence the generalization to a macro.
 * create buffers + slices as needed + views as needed for the buffer of the forward pass.
 */
#ifndef THRNN_LSTM_FORWARD_BUFFERS
#define THRNN_LSTM_FORWARD_BUFFERS() \
    THLongStorage* default_buffer_sizes = THLongStorage_newWithSize(1); \
    default_buffer_sizes->data[0] = nEmbeddings; \
    struct THRNN_(buffer)* rnn_buffer = THRNN_(create_buffer)(buffer, default_buffer_sizes); \
\
    THTensor* ifog_hat_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 4 * nEmbeddings * outputFeatures); \
    THTensor* ifo_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 3 * nEmbeddings * outputFeatures); \
    THTensor* g_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, nEmbeddings * outputFeatures); \
    THTensor* c_buffer_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, nEmbeddings * outputFeatures); \
    THTensor* c_buffer2_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, nEmbeddings * outputFeatures); \
    THTensor* c_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, nEmbeddings * outputFeatures); \
    THTensor* tanh_c_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, nEmbeddings * outputFeatures); \
\
    THTensor* ifo_hat_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 0); \
    THTensor* g_hat_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 0); \
    THTensor* i_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 0); \
    THTensor* f_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 0); \
    THTensor* o_t = THRNN_(add_tensor_to_buffer)(rnn_buffer, 0); \
\
    THRNN_(compile_buffer)(rnn_buffer); \
\
    THTensor_(resize2d)(ifog_hat_t, nEmbeddings, 4*outputFeatures); \
    THTensor_(resize2d)(ifo_t, nEmbeddings, 3*outputFeatures); \
    THTensor_(resize2d)(g_t, nEmbeddings, outputFeatures); \
    THTensor_(resize2d)(c_buffer_t, nEmbeddings, outputFeatures); \
    THTensor_(resize2d)(c_buffer2_t, nEmbeddings, outputFeatures); \
    THTensor_(resize2d)(c_t, nEmbeddings, outputFeatures); \
    THTensor_(resize2d)(tanh_c_t, nEmbeddings, outputFeatures); \
    THTensor_(narrow)(ifo_hat_t, ifog_hat_t, 1, 0, 3*outputFeatures); \
    THTensor_(narrow)(g_hat_t, ifog_hat_t, 1, 3*outputFeatures, outputFeatures); \
    THTensor_(narrow)(i_t, ifo_t, 1, 0, outputFeatures); \
    THTensor_(narrow)(f_t, ifo_t, 1, outputFeatures, outputFeatures); \
    THTensor_(narrow)(o_t, ifo_t, 1, outputFeatures*2, outputFeatures);
#endif

#ifndef THRNN_LSTM_GRU_CELL
#define THRNN_LSTM_GRU_CELL 2
#endif

// input is organized time-first, concatenated,
// in decreasing order of batch size per time step.
// for example, if the batch of sequences is:
// { Tensor(2,100), Tensor(5,100), Tensor(4,100) },
// then the data has to be transposed and sorted and will look like:
// input: torch.Tensor(5+4+2)
// sizes: torch.LongTensor({5,4,2})
void THRNN_(LSTM_updateOutput)(
          THRNNState    *state,
          THTensor      *input,
          THTensor      *inputC,
          THTensor      *inputH,
          THLongTensor  *sizes,
          THTensor      *output,
          THTensor      *weight,
          THTensor      *bias,
          THTensor      *buffer,
          int           inputFeatures,
          int           outputFeatures,
          int           cellType,
          int           train,
          int           delete_buffer)
{
  long sizesDim = THLongTensor_nDimension(sizes);
  long inputDim = THTensor_(nDimension)(input);
  THArgCheck(sizesDim == 1, 2, "sizes must have 1 dimension");
  THArgCheck(inputDim == 2, 1, "input must have 2 dimensions");

  // Retrieve all the dimensions of the problem
  long totalTimeSteps = THLongTensor_size(sizes, 0);
  long nEmbeddings = THTensor_(size)(input, 0);
  long inputSize1 = THTensor_(size)(input, 1);
  long weightSize0 = THTensor_(size)(weight, 0);
  long weightSize1 = THTensor_(size)(weight, 1);
  long* sizesData = THLongTensor_data(sizes);

  // Resize the output
  THTensor_(resize2d)(output, nEmbeddings, outputFeatures);


  // Compute the sum of sizes to
  // further check they're equal to the number of embeddings
  THLongTensor* sizesSum = THLongTensor_new();
  THLongTensor_sum(sizesSum, sizes, 0, 1);
  long ssum = sizesSum->storage->data[0];
  THLongTensor_free(sizesSum);

  // Make sure these inputs are contiguous to accelerate computations
  THArgCheck(THTensor_(isContiguous)(input), 1, "input vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(inputH), 6, "inputH vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(inputC), 5, "inputC vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output), 3, "output vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight matrix must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 5, "bias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(buffer), 8, "buffer tensor must be contiguous");
  THArgCheck(ssum == nEmbeddings, 9, "the sum of sizes must be equal to the number of input embeddings");
  long t;

  if (cellType == THRNN_LSTM_BASIC_LSTM_CELL)
  {

    /*
     * Definitions:
     * w == weights for the fully-connected layer.
     * Weights are concatenated for [input_gate,forget_gate,output_gate,input_tansformation], in this exact order.
     * b == expanded bias
     * i_t == input gate at timestep t
     * f_t == forget gate at timestep t
     * o_t == output gate at timestep t
     * g_t == input transformation at timestep t
     * ifog_hat_t == concatenation of i_t,f_t, o_t and g_t before applying the non-linearities
     * ifo_t == concatenation of i_t,f_t and o_t
     * input_t == input at timestep t
     * w_x == weights for the input part
     * w_h == weights for the hidden part
     *
     * Sequence of computations:
     * 1/ matrix-multiply input for all timesteps by weights for input
     * ifog_hat_{all} = w_x * input_{all} + b
     *
     * 2/ for each timestep t do:
     * ifog_hat_t += w_h * h_t + b
     * ifo_t = sigmoid(ifo_hat_t)
     * g_t = tanh(g_hat_t)
     * c_buffer_t = f_t o c_{t-1}
     * c_buffer2_t = i_t o g_t
     * c_t = c_buffer_t + c_buffer2_t
     * tanh_c_t = tanh(c_t)
     * output_t = o_t o tanh_c_t
     */

    long totalWeightColumns = outputFeatures*4;
    THAssert(totalWeightColumns == weightSize1);

    // Alright, this will create ALL the buffers we need for the forward pass
    // Allocating a big contiguous dude will help us slice intelligently
    // when performing our dear computations.
    // This will also keep track of stuff we need to store for the backward pass.
    THRNN_LSTM_FORWARD_BUFFERS()

    // Initialize a few extra "non-temporal" buffers
    THLongStorage* weight_buffer_sizes = THLongStorage_newWithSize(1);
    weight_buffer_sizes->data[0] = 1;
    struct THRNN_(buffer)* weight_buffer = THRNN_(create_buffer)(NULL, weight_buffer_sizes);
    THTensor* w_h = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* w_x = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* c_tless = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* h_tless = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);

    THRNN_(compile_buffer)(weight_buffer);

    // Slice weights for hidden nodes and inputs
    THTensor_(narrow)(w_h, weight, 0, inputFeatures, outputFeatures);
    THTensor_(narrow)(w_x, weight, 0, 0, inputFeatures);

    // Resize/expand the bias
    THTensor_(resize2d)(bias, 1, THTensor_(size)(bias, 0));
    THRNN_LSTM_SET_STRIDE(bias, 0, 0);
    THRNN_LSTM_SET_SIZE(bias, 0, nEmbeddings);

    // Ok now start computations
    // First, pre-multiply the input with the weights
    // Pre-multiply the input by the weight
    THTensor_(addmm)(ifog_hat_t, 1, bias, 1, input, w_x);

    for (t = 0; t < totalTimeSteps; t++)
    {
      long bsize = sizesData[t];

      // Narrow the buffers along the first dimension
      // to be equal to the batch size for this particular timestep
      // We're not using THTensor_(narrow) for efficiency.
      THRNN_(buffer_set_size)(rnn_buffer, 0, bsize);
      THRNN_LSTM_SET_SIZE(output, 0, bsize);
      THRNN_LSTM_SET_SIZE(bias, 0, bsize);

      if (!t)
      {
        THTensor_(narrow)(c_tless, inputC, 0, 0, sizesData[t]);
        THTensor_(narrow)(h_tless, inputH, 0, 0, sizesData[t]);
      }
      // Add hidden values after mat-mul
      // to the existing input values after matmul (obtained from the pre-multiplication)
      THTensor_(addmm)(ifog_hat_t, 1, ifog_hat_t, 1, h_tless, w_h);

      // Sigmoidize the first 3 slices
      // Non-contiguous operation but heh
      THTensor_(sigmoid)(ifo_t, ifo_hat_t);

      // Tanhize the last slice
      // Non-contiguous operation but heh
      THTensor_(tanh)(g_t, g_hat_t);

      // apply the forget gate
      // Contiguous operation
      THTensor_(cmul)(c_buffer_t, f_t, c_tless);

      // apply the input gate
      // Contiguous operation
      THTensor_(cmul)(c_buffer2_t, i_t, g_t);

      // add the residual cell value to the previous cell
      // and update the cell state
      // Contiguous operation
      THTensor_(cadd)(c_t, c_buffer_t, 1, c_buffer2_t);

      // apply the tanh to the output cell
      // Contiguous operations
      THTensor_(tanh)(tanh_c_t, c_t);

      // apply the output gate
      // and update the hidden state
      // Contiguous operations
      THTensor_(cmul)(output, tanh_c_t, o_t);

      if (t < totalTimeSteps-1)
      {
        THTensor_(narrow)(c_tless, c_t, 0, 0, sizesData[t+1]);
        THTensor_(narrow)(h_tless, output, 0, 0, sizesData[t+1]);
      }

      // After the computations for that timestep are done,
      // increment the offset to switch to the next timestep
      THRNN_(buffer_increment_soffset)(rnn_buffer, 0, bsize);
      THRNN_LSTM_INCREMENT_SOFFSET(output, 0, bsize);
    }
    // Get all the buffers back to their original states
    THRNN_(buffer_decrement_soffset)(rnn_buffer, 0, nEmbeddings);
    THRNN_(buffer_set_size)(rnn_buffer, 0, nEmbeddings);
    THRNN_LSTM_DECREMENT_SOFFSET(output, 0, nEmbeddings);

    // Reshape the bias to its original shape
    THRNN_LSTM_SET_SIZE(bias, 0, 1);
    THTensor_(resize1d)(bias, THTensor_(size)(bias, 1));
    THRNN_LSTM_SET_STRIDE(bias, 0, 1);

    // Reshape the hidden and cell values to the max batch size
    // resize the output back to its total size
    THRNN_LSTM_SET_SIZE(output, 0, nEmbeddings);

    THRNN_(delete_buffer)(rnn_buffer, train?0:delete_buffer);
    THRNN_(delete_buffer)(weight_buffer, 0);

  }
  return;
}




void THRNN_(LSTM_backward)(
          THRNNState    *state,
          THTensor      *input,
          THTensor      *inputC,
          THTensor      *inputH,
          THLongTensor  *sizes,
          THTensor      *gradOutput,
          THTensor      *gradInput,
          THTensor      *gradInputH,
          THTensor      *gradInputC,
          THTensor      *weight,
          THTensor      *bias,
          THTensor      *buffer,
          THTensor      *weightBuffer,
          THTensor      *gradInputBuffer,
          THTensor      *gradWeight,
          THTensor      *gradBias,
          THTensor      *output,
          real          scale,
          int           last,
          int           inputFeatures,
          int           outputFeatures,
          int           cellType,
          int           delete_buffer)
{
  long sizesDim = THLongTensor_nDimension(sizes);
  long inputDim = THTensor_(nDimension)(input);
  THArgCheck(sizesDim == 1, 2, "sizes must have 1 dimension");
  THArgCheck(inputDim == 2, 1, "input must have 2 dimensions");
  // Retrieve all the dimensions of the problem
  long totalTimeSteps = THLongTensor_size(sizes, 0);
  long nEmbeddings = THTensor_(size)(input, 0);
  long inputSize1 = THTensor_(size)(input, 1);
  long weightSize0 = THTensor_(size)(weight, 0);
  long weightSize1 = THTensor_(size)(weight, 1);
  long* sizesData = THLongTensor_data(sizes);

  // Resize the output
  THTensor_(resize2d)(gradInput, nEmbeddings, inputSize1);
  THTensor_(resize2d)(gradWeight, weightSize0, weightSize1);
  THTensor_(resize1d)(gradBias, 4*outputFeatures);


  THLongTensor* sizesSum = THLongTensor_new();
  THLongTensor_sum(sizesSum, sizes, 0, 1);
  long ssum = sizesSum->storage->data[0];
  THLongTensor_free(sizesSum);

  // Make sure these inputs are contiguous to accelerate computations
  THArgCheck(THTensor_(isContiguous)(input), 1, "input vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(inputH), 6, "inputH vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(inputC), 5, "inputC vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output), 3, "output vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight matrix must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 5, "bias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(buffer), 8, "buffer tensor must be contiguous");
  THArgCheck(ssum == nEmbeddings, 9, "the sum of sizes must be equal to the number of input embeddings");
  THArgCheck(THTensor_(nDimension)(gradOutput) == THTensor_(nDimension)(output), 3, "output and gradOutput do not have the same # of dimensions");
  THArgCheck(THTensor_(size)(gradOutput, 1) == THTensor_(size)(output, 1), 3, "when the output type is 'last', gradOutput and output must have the same size for the 2nd dimension");

  long t;

  if (cellType == THRNN_LSTM_BASIC_LSTM_CELL)
  {
    // Resize the buffer
    long totalWeightColumns = outputFeatures*4;
    THAssert(totalWeightColumns == weightSize1);

    // Alright, this buffer is going to contain ALL the buffers we need.
    // This will also keep track of stuff we need to store for the backward pass.
    THRNN_LSTM_FORWARD_BUFFERS()

    THLongStorage* gi_default_buffer_sizes = THLongStorage_newWithSize(1);
    gi_default_buffer_sizes->data[0] = nEmbeddings;
    struct THRNN_(buffer)* gi_rnn_buffer = THRNN_(create_buffer)(gradInputBuffer, gi_default_buffer_sizes);

    THTensor* difog_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 4 * nEmbeddings * outputFeatures);
    THTensor* dc_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, nEmbeddings * outputFeatures);
    THTensor* dtanh_c_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, nEmbeddings * outputFeatures);
    THTensor* ones = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, nEmbeddings * outputFeatures);
    THTensor* x_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, nEmbeddings * inputFeatures);
    THTensor* dh_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, nEmbeddings * outputFeatures);

    THTensor* difo_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 0);
    THTensor* di_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 0);
    THTensor* df_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 0);
    THTensor* do_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 0);
    THTensor* dg_t = THRNN_(add_tensor_to_buffer)(gi_rnn_buffer, 0);

    THRNN_(compile_buffer)(gi_rnn_buffer);

    THTensor_(resize2d)(difog_t, nEmbeddings, 4*outputFeatures);
    THTensor_(resize2d)(dc_t, nEmbeddings, outputFeatures);
    THTensor_(resize2d)(dtanh_c_t, nEmbeddings, outputFeatures);
    THTensor_(resize2d)(ones, nEmbeddings, outputFeatures);
    THTensor_(resize2d)(dh_t, nEmbeddings, outputFeatures);

    THTensor_(narrow)(difo_t, difog_t, 1, 0, 3*outputFeatures);
    THTensor_(narrow)(di_t, difog_t, 1, 0, outputFeatures);
    THTensor_(narrow)(df_t, difog_t, 1, outputFeatures, outputFeatures);
    THTensor_(narrow)(do_t, difog_t, 1, 2*outputFeatures, outputFeatures);
    THTensor_(narrow)(dg_t, difog_t, 1, 3*outputFeatures, outputFeatures);
    THTensor_(fill)(ones, 1);


    THLongStorage* weight_buffer_sizes = THLongStorage_newWithSize(1);
    weight_buffer_sizes->data[0] = 1;
    struct THRNN_(buffer)* weight_buffer = THRNN_(create_buffer)(weightBuffer, weight_buffer_sizes);

    THTensor* w_h = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* w_x = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* w_hT = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* w_xT = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* grad_weight_h = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* grad_weight_x = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* h_tlessT = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* x_tT = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* c_tless = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* h_tless = THRNN_(add_tensor_to_buffer)(weight_buffer, 0);
    THTensor* grad_bias_buffer = THRNN_(add_tensor_to_buffer)(weight_buffer, 4*outputFeatures);
    THTensor* grad_input_c = THRNN_(add_tensor_to_buffer)(weight_buffer, sizesData[0] * outputFeatures);
    THTensor* grad_input_h = THRNN_(add_tensor_to_buffer)(weight_buffer, sizesData[0] * outputFeatures);

    THRNN_(compile_buffer)(weight_buffer);

    THTensor_(narrow)(w_h, weight, 0, inputFeatures, outputFeatures);
    THTensor_(narrow)(w_x, weight, 0, 0, inputFeatures);
    THTensor_(narrow)(grad_weight_h, gradWeight, 0, inputFeatures, outputFeatures);
    THTensor_(narrow)(grad_weight_x, gradWeight, 0, 0, inputFeatures);
    THTensor_(transpose)(w_hT, w_h, 0, 1);
    THTensor_(transpose)(w_xT, w_x, 0, 1);
    THTensor_(resize2d)(grad_input_c, sizesData[0], outputFeatures);
    THTensor_(resize2d)(grad_input_h, sizesData[0], outputFeatures);
    THTensor_(resize2d)(grad_bias_buffer, 1, 4*outputFeatures);

    // Increment all the buffer up to the last timestep
    THRNN_(buffer_increment_soffset)(rnn_buffer, 0, nEmbeddings);
    THRNN_(buffer_increment_soffset)(gi_rnn_buffer, 0, nEmbeddings);
    THRNN_LSTM_INCREMENT_SOFFSET(output, 0, nEmbeddings);

    // If the output type is not 'last',
    // then it means that gradOutput has the same size as output.
    // Otherwise it must be of size batchSize x outputSize
    if (!last)
    {
      THArgCheck(THTensor_(size)(gradOutput, 0) == THTensor_(size)(output, 0), 3, "when the output type is not 'last', gradOutput and output must have the same size for the first dimension");
      THRNN_LSTM_INCREMENT_SOFFSET(gradOutput, 0, nEmbeddings);
    }
    else
    {
      THArgCheck(THTensor_(size)(gradOutput, 0) == sizesData[0], 3, "when the output type is 'last', the size of gradOutput for the first dimension must be equal to the batch size");
    }

    THTensor_(copy)(grad_input_c, gradInputC);
    THTensor_(copy)(grad_input_h, gradInputH);

    for (t = totalTimeSteps-1; t >= 0; t--)
    {
      long bsize = sizesData[t];

      THRNN_(buffer_decrement_soffset)(rnn_buffer, 0, bsize);
      THRNN_(buffer_decrement_soffset)(gi_rnn_buffer, 0, bsize);
      THRNN_LSTM_DECREMENT_SOFFSET(output, 0, bsize);

      // Narrow the buffers along the first dimension
      // We're not using THTensor_(narrow) for efficiency.
      THRNN_(buffer_set_size)(rnn_buffer, 0, bsize);
      THRNN_(buffer_set_size)(gi_rnn_buffer, 0, bsize);
      THRNN_LSTM_SET_SIZE(grad_input_c, 0, bsize);
      THRNN_LSTM_SET_SIZE(grad_input_h, 0, bsize);
      THRNN_LSTM_SET_SIZE(output, 0, bsize);

      if (!t)
      {
        THTensor_(narrow)(c_tless, inputC, 0, 0, sizesData[t]);
        THTensor_(narrow)(h_tless, inputH, 0, 0, sizesData[t]);
      }
      else
      {
        THRNN_LSTM_DECREMENT_SOFFSET(output, 0, sizesData[t-1]);
        THRNN_LSTM_DECREMENT_SOFFSET(c_t, 0, sizesData[t-1]);
        THTensor_(narrow)(h_tless, output, 0, 0, sizesData[t]);
        THTensor_(narrow)(c_tless, c_t, 0, 0, sizesData[t]);
        THRNN_LSTM_INCREMENT_SOFFSET(output, 0, sizesData[t-1]);
        THRNN_LSTM_INCREMENT_SOFFSET(c_t, 0, sizesData[t-1]);
      }
      // If the output type is not 'last',
      // We need to decrement the offset for gradOutput as well,
      // and set the size
      if (!last)
      {
        THRNN_LSTM_DECREMENT_SOFFSET(gradOutput, 0, bsize);
        THRNN_LSTM_SET_SIZE(gradOutput, 0, bsize);
      }
      // If we are at the last timestep, copy gradInputH and add gradOutput directly to dh_t,
      // otherwise add it if the output type is not last.
      // For all timesteps != last_timestep,
      // accumulate the gradients from the next timestep
      THTensor_(copy)(dh_t, grad_input_h);
      if (t == totalTimeSteps-1 || !last)
      {
        THTensor_(cadd)(dh_t, dh_t, 1, gradOutput);
      }
      // Compute do_t = dh_t o tanh(c_t)
      THTensor_(cmul)(do_t, dh_t, tanh_c_t);

      // Compute dc_t += (1-tanh^2(c_t)) o o_t o dh_t
      THTensor_(cmul)(dtanh_c_t, tanh_c_t, tanh_c_t);
      THTensor_(cadd)(dc_t, ones, -1, dtanh_c_t);
      THTensor_(cmul)(dc_t, dc_t, dh_t);
      THTensor_(cmul)(dc_t, dc_t, o_t);
      THTensor_(cadd)(grad_input_c, grad_input_c, 1, dc_t);

      // Now compute di_t = dc_t o g_t
      THTensor_(cmul)(di_t, grad_input_c, g_t);

      // Now compute df_t = dc_t o c(t-1)
      THTensor_(cmul)(df_t, grad_input_c, c_tless);

      // Now compute dg_t = dc_t o i_t
      THTensor_(cmul)(dg_t, grad_input_c, i_t);
      THTensor_(cmul)(grad_input_c, grad_input_c, f_t);


      // Compute di_t = di_t o i_t o (1-i_t)
      // Compute df_t = df_t o f_t o (1-f_t)
      // Compute do_t = df_t o f_t o (1-o_t)
      // Compute dg_t = dg_t o (1-tanh^2(g_t))
      THTensor_(cadd)(dc_t, ones, -1, i_t);
      THTensor_(cmul)(di_t, di_t, dc_t);
      THTensor_(cmul)(di_t, di_t, i_t);

      THTensor_(cadd)(dc_t, ones, -1, f_t);
      THTensor_(cmul)(df_t, df_t, dc_t);
      THTensor_(cmul)(df_t, df_t, f_t);

      THTensor_(cadd)(dc_t, ones, -1, o_t);
      THTensor_(cmul)(do_t, do_t, dc_t);
      THTensor_(cmul)(do_t, do_t, o_t);

      THTensor_(cmul)(dc_t, g_t, g_t);
      THTensor_(cadd)(dtanh_c_t, ones, -1, dc_t);
      THTensor_(cmul)(dg_t, dg_t, dtanh_c_t);


      THTensor_(transpose)(h_tlessT, h_tless, 0, 1);

      // Accumulate the gradient wrt to the bias
      THTensor_(sum)(grad_bias_buffer, difog_t, 0, 1);
      THTensor_(cadd)(gradBias, gradBias, scale, grad_bias_buffer);

      // Accumulate the gradient wrt to the weight, for hidden nodes only
      THTensor_(addmm)(grad_weight_h, 1, grad_weight_h, scale, h_tlessT, difog_t);

      // Compute the gradient wrt the input
      THTensor_(zero)(grad_input_h);
      THTensor_(addmm)(grad_input_h, 0, grad_input_h, 1, difog_t, w_hT);
    }

    THRNN_(buffer_set_size)(rnn_buffer, 0, nEmbeddings);
    THRNN_(buffer_set_size)(gi_rnn_buffer, 0, nEmbeddings);

    THTensor_(transpose)(x_tT, input, 0, 1);
    THTensor_(addmm)(grad_weight_x, 1, grad_weight_x, scale, x_tT, difog_t);
    THTensor_(zero)(gradInput);
    THTensor_(addmm)(gradInput, 0, gradInput, 1, difog_t, w_xT);

    THRNN_LSTM_SET_SIZE(grad_input_h, 0, sizesData[0]);
    THRNN_LSTM_SET_SIZE(grad_input_c, 0, sizesData[0]);
    THRNN_LSTM_SET_SIZE(output, 0, nEmbeddings);
    THRNN_LSTM_SET_SIZE(c_tless, 0, sizesData[0]);
    THRNN_LSTM_SET_SIZE(h_tless, 0, sizesData[0]);
    if (!last)
    {
      THRNN_LSTM_SET_SIZE(gradOutput, 0, nEmbeddings);
    }
    THRNN_(delete_buffer)(rnn_buffer, delete_buffer);
    THRNN_(delete_buffer)(gi_rnn_buffer, delete_buffer);
    THRNN_(delete_buffer)(weight_buffer, delete_buffer);
  }
  return;
}
#endif
