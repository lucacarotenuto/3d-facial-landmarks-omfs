#ifndef MACROS_H
#define MACROS_H

#define DEBUG 1

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_DIMENSIONS(x) AT_ASSERTM((x).dim() == 2, #x " must be two-dimensional")
#define CHECK_DTYPE(x, t) AT_ASSERTM((x).dtype() == t, #x " must have torch.int32 dtype")
#define CHECK_INPUT(x, t) CHECK_DIMENSIONS(x); CHECK_DTYPE(x, t)

#define NUM_BLOCKS 64
#define BLOCK_SIZE 256

#define MAX_NHOOD 30


#endif