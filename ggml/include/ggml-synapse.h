#pragma once

#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_SYNAPSE_NAME "SYNAPSE"

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_synapse_init(int device);

GGML_BACKEND_API bool ggml_backend_is_synapse(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_buffer_is_synapse(ggml_backend_buffer_t buffer);

GGML_BACKEND_API bool ggml_backend_buft_is_synapse(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API int ggml_backend_synapse_get_device_count(void);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_synapse_buffer_type(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_synapse_reg(void);

#ifdef __cplusplus
}
#endif
