#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THRNN.h"
#else

TH_API void THRNN_(LSTM_updateOutput)(
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
          int           delete_buffer);

TH_API void THRNN_(LSTM_backward)(
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
          THTensor      *gradInputBuffer,
          THTensor      *gradWeight,
          THTensor      *gradBias,
          THTensor      *output,
          real          scale,
          int           last,
          int           inputFeatures,
          int           outputFeatures,
          int           cellType,
          int           delete_buffer);
#endif
