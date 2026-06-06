#ifndef NAUTILUS_TVM_C_API_H
#define NAUTILUS_TVM_C_API_H

/*
 * tvm_c_api.h — Stable C ABI for Apache TVM MetaSchedule.
 *
 * Used for the autotune step: take a TIR module, run MetaSchedule,
 * get back a best-config.
 *
 * The C ABI isolates Nautilus from TVM's fast-moving C++ API.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nautilus_tir_module_s nautilus_tir_module_t;
typedef struct nautilus_tuning_record_s nautilus_tuning_record_t;

/* Sharding strategy hint. */
typedef enum {
    NAUTILUS_TUNING_AUTO = 0,
    NAUTILUS_TUNING_RL_EVOLUTIONARY = 1,
    NAUTILUS_TUNING_RL_GRADIENT = 2,
    NAUTILUS_TUNING_XGBOOST_COST_MODEL = 3,
} nautilus_tuning_strategy_t;

/*
 * Parse a TIR module from text. The text is the TVMScript or TIR
 * textual representation.
 */
int nautilus_tir_parse(
    const char* text,
    size_t text_len,
    const char* target,         /* e.g. "nvidia/nvidia-h100" */
    nautilus_tir_module_t** out
);

void nautilus_tir_release(nautilus_tir_module_t* mod);

/*
 * Run MetaSchedule on the TIR module. This is the heavy operation.
 * May run for minutes; the wrapper exposes a timeout.
 *
 * @param mod             Module to tune
 * @param max_trials      Maximum number of trials
 * @param num_trials_per_iter  Trials per evolutionary iteration
 * @param strategy        Tuning strategy
 * @param timeout_seconds Wall-clock timeout
 * @param out_record      Receives the best tuning record
 * @return                0 on success, NAUTILUS_ERR_TUNING_TIMEOUT
 *                        if exceeded timeout, NAUTILUS_ERR_NO_RECORDS
 *                        if MetaSchedule found no valid config
 */
int nautilus_tune(
    nautilus_tir_module_t* mod,
    int max_trials,
    int num_trials_per_iter,
    nautilus_tuning_strategy_t strategy,
    int timeout_seconds,
    nautilus_tuning_record_t** out_record
);

void nautilus_tuning_record_release(nautilus_tuning_record_t* rec);

/*
 * Extract a tuning parameter (BLOCK_M, num_warps, etc.) from a record.
 * Returns 0 on success, NAUTILUS_ERR_INVALID_ARG if the param is missing.
 */
int nautilus_record_get_int(
    nautilus_tuning_record_t* rec,
    const char* name,
    int64_t* out_value
);

/* Error codes */
#define NAUTILUS_TUNING_OK              0
#define NAUTILUS_TUNING_ERR_GENERIC    -1
#define NAUTILUS_TUNING_ERR_PARSE      -2
#define NAUTILUS_TUNING_ERR_TIMEOUT    -3
#define NAUTILUS_TUNING_ERR_NO_RECORDS -4
#define NAUTILUS_TUNING_ERR_BACKEND    -5

const char* nautilus_tvm_last_error_message(void);
const char* nautilus_tvm_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NAUTILUS_TVM_C_API_H */
