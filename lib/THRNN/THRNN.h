#ifndef THRNN_H
#define THRNN_H

#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THRNN_(NAME) TH_CONCAT_3(THRNN_, Real, NAME)

typedef long THIndex_t;
typedef int THInteger_t;
typedef void THRNNState;

#define THRNN_resizeAs_indices(I1, I2)                    \
  THLongStorage *size2 = THIndexTensor_(newSizeOf)(I2);  \
  if (!THTensor_(isSize)(I1, size2))                     \
  { \
    THTensor_(resize)(I1, size2, NULL);                  \
  } \
  THLongStorage_free(size2);

#include "generic/THRNN.h"
#include <THGenerateFloatTypes.h>

#endif
