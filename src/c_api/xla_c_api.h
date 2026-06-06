#ifndef NAUTILUS_XLA_C_API_H
#define NAUTILUS_XLA_C_API_H

/*
 * xla_c_api.h — Stable C ABI for OpenXLA / StableHLO / GSPMD.
 *
 * Used by Phase 3: PyTorch model -> StableHLO -> GSPMD sharding.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nautilus_stablehlo_s nautilus_stablehlo_t;
typedef struct nautilus_sharding_spec_s nautilus_sharding_spec_t;
typedef struct nautilus_mesh_s nautilus_mesh_t;

/* Sharding strategy. */
typedef enum {
    NAUTILUS_SHARD_AUTO = 0,         /* Let GSPMD decide */
    NAUTILUS_SHARD_REPLICATED = 1,   /* Replicate tensor */
    NAUTILUS_SHARD_DATA_PARALLEL = 2,
    NAUTILUS_SHARD_MODEL_PARALLEL = 3,
    NAUTILUS_SHARD_TENSOR_PARALLEL = 4,
} nautilus_shard_strategy_t;

/*
 * Build a StableHLO module from a TorchFX graph (serialized as JSON
 * or as the text-form of a torch.export.ExportedProgram).
 */
int nautilus_stablehlo_from_fx(
    const char* fx_graph_json,
    size_t json_len,
    const char* function_name,
    nautilus_stablehlo_t** out
);

void nautilus_stablehlo_release(nautilus_stablehlo_t* mod);
int nautilus_stablehlo_get_mlir_text(
    nautilus_stablehlo_t* mod,
    const char** out_text,
    size_t* out_len
);

/*
 * Build a mesh: a logical device grid of shape (a, b, c, ...).
 */
int nautilus_mesh_create(
    const int64_t* axes,
    size_t n_axes,
    nautilus_mesh_t** out
);

void nautilus_mesh_release(nautilus_mesh_t* mesh);

/*
 * Run GSPMD on a StableHLO module with the given mesh. Returns a
 * sharding spec describing how each tensor is partitioned.
 */
int nautilus_gspmd_shard(
    nautilus_stablehlo_t* mod,
    nautilus_mesh_t* mesh,
    nautilus_shard_strategy_t strategy,
    int timeout_seconds,
    nautilus_sharding_spec_t** out_spec
);

void nautilus_sharding_spec_release(nautilus_sharding_spec_t* spec);

/* Error codes */
#define NAUTILUS_XLA_OK                 0
#define NAUTILUS_XLA_ERR_GENERIC       -1
#define NAUTILUS_XLA_ERR_FX_PARSE      -2
#define NAUTILUS_XLA_ERR_EXPORT        -3
#define NAUTILUS_XLA_ERR_MESH          -4
#define NAUTILUS_XLA_ERR_GSPMD         -5
#define NAUTILUS_XLA_ERR_TIMEOUT       -6

const char* nautilus_xla_last_error_message(void);
const char* nautilus_xla_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NAUTILUS_XLA_C_API_H */
