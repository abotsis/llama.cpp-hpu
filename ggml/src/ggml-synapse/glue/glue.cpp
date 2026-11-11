// Graph-compiler glue for the ggml-synapse TPC kernels (GC_KERNEL_PATH plugin).
// Every kernel takes the number of work members as scalar param 0; the index space is
// [members] and all tensors are allRequired, so the compiler never slices them and each kernel
// splits its own work.
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace tpc_lib_api;

#define K(name)                                                     \
    extern "C" const unsigned char _binary_##name##_bin_start[];    \
    extern "C" const unsigned char _binary_##name##_bin_end[];
#include "kernel_table.h"
#undef K

struct kdef {
    const char *          name;
    const unsigned char * begin;
    const unsigned char * end;
};

static const kdef k_kernels[] = {
#define K(name) { #name, _binary_##name##_bin_start, _binary_##name##_bin_end },
#include "kernel_table.h"
#undef K
};

static const unsigned k_count = sizeof(k_kernels) / sizeof(k_kernels[0]);

static void guid_of(const kdef & k, char * out) {
    snprintf(out, MAX_NODE_NAME, "%s_f32", k.name);
}

extern "C" {

__attribute__((visibility("default")))
GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t * kernelCount, GuidInfo * guids) {
    if (deviceId != DEVICE_ID_GAUDI2) {
        if (kernelCount) {
            *kernelCount = 0;
        }
        return GLUE_SUCCESS;
    }
    if (guids) {
        for (unsigned i = 0; i < k_count && i < *kernelCount; i++) {
            memset(&guids[i], 0, sizeof(GuidInfo));
            guid_of(k_kernels[i], guids[i].name);
        }
    }
    if (kernelCount) {
        *kernelCount = k_count;
    }
    return GLUE_SUCCESS;
}

__attribute__((visibility("default")))
GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams * p, HabanaKernelInstantiation * inst) {
    const kdef * k = nullptr;
    char g[MAX_NODE_NAME];
    for (unsigned i = 0; i < k_count; i++) {
        guid_of(k_kernels[i], g);
        if (strcmp(p->guid.name, g) == 0) {
            k = &k_kernels[i];
            break;
        }
    }
    if (!k) {
        return GLUE_NODE_NOT_FOUND;
    }
    if (p->nodeParams.nodeParams == nullptr || p->nodeParams.nodeParamsSize < 4) {
        return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    int32_t nmem = *(const int32_t *) p->nodeParams.nodeParams;
    if (nmem < 1) {
        nmem = 1;
    }
    inst->indexSpaceRank = 1;
    for (int d = 0; d < 5; d++) {
        inst->indexSpaceGeometry[d] = d == 0 ? (uint64_t) nmem : 1;
    }
    for (uint32_t t = 0; t < p->inputTensorNr; t++) {
        TensorAccessPattern & ap = inst->inputTensorAccessPattern[t];
        memset(&ap, 0, sizeof(ap));
        ap.allRequired = 1;
        for (uint32_t d = 0; d < MAX_TENSOR_DIM; d++) {
            ap.mapping[d].end_b = float(p->inputTensors[t].geometry.maxSizes[d]) - 1;
        }
    }
    for (uint32_t t = 0; t < p->outputTensorNr; t++) {
        TensorAccessPattern & ap = inst->outputTensorAccessPattern[t];
        memset(&ap, 0, sizeof(ap));
        ap.allRequired = 1;
        for (uint32_t d = 0; d < MAX_TENSOR_DIM; d++) {
            ap.mapping[d].end_b = float(p->outputTensors[t].geometry.maxSizes[d]) - 1;
        }
    }
    // paged dequantization (syn_qdqp_*): index space [row blocks, superblocks, pages] over its output [M, K, P]
    // (params: members, K, M, superblock k-steps), output access pattern affine in it; inputs whole.
    // GGML_SYNAPSE_NO_AP=1: output allRequired.
    if (strncmp(p->guid.name, "syn_qdqp_", 9) == 0) {
        if (p->nodeParams.nodeParamsSize < 16 || p->outputTensorNr < 1) {
            return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        }
        const int32_t * prm = (const int32_t *) p->nodeParams.nodeParams;
        const uint64_t K = (uint64_t) prm[1], M = (uint64_t) prm[2], SBk = (uint64_t) prm[3];
        const uint64_t P = p->outputTensors[0].geometry.maxSizes[2];
        if (SBk == 0 || K % SBk != 0) {
            return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        }
        inst->indexSpaceRank = 3;
        inst->indexSpaceGeometry[0] = (M + 127) / 128;
        inst->indexSpaceGeometry[1] = K / SBk;
        inst->indexSpaceGeometry[2] = P;
        if (!getenv("GGML_SYNAPSE_NO_AP")) {
            TensorAccessPattern & o = inst->outputTensorAccessPattern[0];
            memset(&o, 0, sizeof(o));
            o.allRequired = 0;
            o.mapping[0].indexSpaceDim = 0; o.mapping[0].a = 128;        o.mapping[0].start_b = 0; o.mapping[0].end_b = 127;
            o.mapping[1].indexSpaceDim = 1; o.mapping[1].a = float(SBk); o.mapping[1].start_b = 0; o.mapping[1].end_b = float(SBk) - 1;
            o.mapping[2].indexSpaceDim = 2; o.mapping[2].a = 1;          o.mapping[2].start_b = 0; o.mapping[2].end_b = 0;
            for (uint32_t t = 0; t < p->inputTensorNr; t++) {
                TensorAccessPattern & ap = inst->inputTensorAccessPattern[t];
                ap.allRequired = 0;   // whole range of every dim (the expert is chosen at run time)
            }
        }
    }
    uint32_t nwords = p->nodeParams.nodeParamsSize / 4;
    if (nwords > MAX_SCALAR_PARAMS) {
        return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    memcpy(inst->kernel.scalarParams, p->nodeParams.nodeParams, nwords * 4);
    inst->kernel.paramsNr = nwords;

    const uint32_t len = (uint32_t) (k->end - k->begin);
    const uint32_t given = inst->kernel.elfSize;
    inst->kernel.elfSize = len;
    if (given < len) {
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    memcpy(inst->kernel.kernelElf, k->begin, len);
    return GLUE_SUCCESS;
}

__attribute__((visibility("default")))
GlueCodeReturn GetShapeInference(DeviceId, ShapeInferenceParams *, ShapeInferenceOutput *) {
    return GLUE_SUCCESS;
}

__attribute__((visibility("default")))
uint64_t GetLibVersion() {
    return 2;
}

} // extern "C"
