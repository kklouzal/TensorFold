#!/bin/bash
# Personal deployment ceilings only. The engine still proves actual capacity.
# Sourcing defines two adapters; call tf_v100_personal_profile before constructing
# serving/benchmark argv and tf_v100_production_launch on the constructed argv.

_tf_v100_positive() {
  [[ $2 =~ ^0*[1-9][0-9]*$ ]] || {
    printf '%s must be a positive decimal integer\n' "$1" >&2
    return 2
  }
}

_tf_v100_ceiling() {
  local name=$1 maximum=$2 value=${!1:-}
  if [[ -z $value || $value == auto ]]; then
    value=$maximum
  else
    _tf_v100_positive "$name" "$value" || return
    value=${value#"${value%%[!0]*}"}
    # Compare bounded decimal text before Bash arithmetic (no overflow).
    if (( ${#value} > ${#maximum} )) ||
       { (( ${#value} == ${#maximum} )) && [[ $value > $maximum ]]; }; then
      printf '%s exceeds the personal V100 ceiling %s\n' "$name" "$maximum" >&2
      return 2
    fi
  fi
  printf -v "$name" '%s' "$value"
  export "$name"
}

tf_v100_personal_profile() {
  local ceiling effective pending http waiting
  TF_V100_PERSONAL_CAP=
  case ${TF_KV_DTYPE:-fp16} in
    int8) ceiling=24 ;;
    rotorquant-iso4-sm70-v1|rotorquant-iso64-norm4-h64) ceiling=32 ;;
    *) return 0 ;;  # No invented ceiling/default for another codec/profile.
  esac
  _tf_v100_ceiling TF_PARALLEL "$ceiling" || return
  _tf_v100_ceiling TF_KV_FULL_WINDOWS "$ceiling" || return
  effective=$TF_PARALLEL
  (( TF_KV_FULL_WINDOWS >= effective )) || effective=$TF_KV_FULL_WINDOWS
  # Default P keeps one bounded FIFO queue as deep as the active windows
  # (2 x effective): requests beyond the windows wait instead of a 503.
  pending=${TF_MAX_PENDING_REQUESTS:-$((2 * effective))}
  _tf_v100_positive TF_MAX_PENDING_REQUESTS "$pending" || return
  # Reuse the exact decimal ceiling validation; P may exceed the GPU windows
  # for bounded waiting, but never twice this personal profile's ceiling.
  TF_MAX_PENDING_REQUESTS=$pending
  _tf_v100_ceiling TF_MAX_PENDING_REQUESTS "$((2 * ceiling))" || return
  http=${TF_MAX_HTTP_CONNECTIONS:-$((TF_MAX_PENDING_REQUESTS + 1))}
  _tf_v100_positive TF_MAX_HTTP_CONNECTIONS "$http" || return
  TF_MAX_HTTP_CONNECTIONS=$http
  TF_CONTEXT=${TF_CONTEXT:-262144}
  TF_DRAFTS=${TF_DRAFTS:-4}
  waiting=${TF_MAX_WAITING_INPUT_BYTES:-}
  if [[ -z $waiting && $TF_CONTEXT =~ ^0*262144$ && $TF_DRAFTS =~ ^0*4$ &&
        ${TF_INDEXER_HISTORY:-gpu} == gpu &&
        ( ${TF_DRAFT_METHOD:-auto} == mtp || ${TF_DRAFT_METHOD:-auto} == auto ) ]]; then
    # Captured SC117/pinned-CPython full-context input claim F=9,437,212.
    # Other geometry/methods require their existing explicit input authority.
    waiting=$((TF_MAX_PENDING_REQUESTS * 9437212))
  fi
  if [[ -n $waiting ]]; then
    _tf_v100_positive TF_MAX_WAITING_INPUT_BYTES "$waiting" || return
  fi
  TF_MAX_WAITING_INPUT_BYTES=$waiting
  TF_V100_PERSONAL_CAP=$ceiling
  export TF_CONTEXT TF_DRAFTS TF_MAX_HTTP_CONNECTIONS TF_MAX_WAITING_INPUT_BYTES TF_V100_PERSONAL_CAP
}

_tf_v100_refuse() {
  printf '%s\n' "$1" >&2
  return 2
}

_tf_v100_manifest() {
  [[ -r $3 ]] ||
    _tf_v100_refuse "no tracked preparation packet $3; set TF_EXECUTION_PREPARATION or TF_V100_LAUNCH=reference" || return
  /opt/venv/bin/python /src/tools/v100_execution_manifest.py manifest --target "$1" \
    --drafter "$2" --preparation "$3" --out "$4" >&2
}

# Production SC117 launch (Compose default, TF_V100_LAUNCH=production). Rewrites the
# ordinary launcher argv in the caller's `args` array: the owned-routed parent
# (native ETP4 experts) replaces launch_cuda_tp.py, and the execution-profile
# manifest (native owner families, prepared linears, owned KV windows and the
# exact-graph catalogue, all captured at startup) is generated from the tracked
# preparation packet into the tmpfs runtime directory. Generation reads only the
# GGUF/drafter headers; nothing prepared is cached on disk.
tf_v100_production_launch() {
  local i split=-1 target='' drafter='' packet rows base
  local runtime=/run/tensorfold-v100/runtime
  local packets=/src/deploy/v100/execution-profiles
  [[ ${TF_THINKING:-on} == on || ${TF_THINKING:-on} == off ]] ||
    _tf_v100_refuse 'TF_THINKING must be on or off' || return
  [[ ${TF_INDEXER_HISTORY:-gpu} == gpu ]] ||
    _tf_v100_refuse 'production launch requires TF_INDEXER_HISTORY=gpu; set TF_V100_LAUNCH=reference otherwise' || return
  case ${TF_DRAFT_METHOD:-auto} in
    auto|mtp) ;;
    *) _tf_v100_refuse 'production launch requires the stock MTP drafter; set TF_V100_LAUNCH=reference otherwise' || return ;;
  esac
  [[ ${TF_EXPERT_TENSOR_PARALLEL_SIZE:-4} == 4 ]] ||
    _tf_v100_refuse 'production launch requires TF_EXPERT_TENSOR_PARALLEL_SIZE=4' || return
  [[ ${TF_PARALLEL:-} =~ ^[1-9][0-9]*$ && ${TF_DRAFTS:-} =~ ^[1-9][0-9]*$ ]] ||
    _tf_v100_refuse 'production launch requires numeric TF_PARALLEL and TF_DRAFTS (a personal KV profile resolves them)' || return
  # Only the personal profiles' codecs (INT8 and the owned-path Q4 formats) run the native owner families the
  # manifest selects; any other codec would otherwise fail later, inside engine startup.
  [[ -n ${TF_V100_PERSONAL_CAP:-} ]] ||
    _tf_v100_refuse "production launch requires a personal-profile KV codec (int8, rotorquant-iso4-sm70-v1 or rotorquant-iso64-norm4-h64), not ${TF_KV_DTYPE:-fp16}; set TF_V100_LAUNCH=reference otherwise" || return
  packet=${TF_EXECUTION_PREPARATION:-$packets/sc117-tp8etp4-mtp${TF_DRAFTS}-p${TF_PARALLEL}.preparation.json}
  for ((i=0; i<${#args[@]}; i++)); do
    if (( split < 0 )); then
      [[ ${args[i]} == -- ]] && split=$i
    elif [[ -z $target ]]; then
      target=${args[i]}
    elif [[ ${args[i]} == --drafter ]]; then
      drafter=${args[i+1]}
    elif [[ ${args[i]} == --draft-method ]]; then
      args[i+1]=mtp  # The owned parent sizes its rows for explicit stock MTP; auto resolves to it here.
    fi
  done
  [[ ${args[0]} == /src/tools/launch_cuda_tp.py && $split -gt 0 && -n $target && -n $drafter ]] ||
    _tf_v100_refuse 'production launch expects the Compose launch_cuda_tp.py argv with a target and --drafter' || return
  rows=$(( TF_PARALLEL * (TF_DRAFTS + 1) ))
  base=$(mktemp -d "$runtime/launch.XXXXXXXX") || return
  _tf_v100_manifest "$target" "$drafter" "$packet" "$base/profile.json" || return
  args=(/src/tools/v100_owned_qualification.py "${args[@]:1:split-1}" --out "$base/receipts"
        --execution-rows "$rows" --grouped-gate-up-rows 4 "${args[@]:split}"
        --execution-profile-manifest "$base/profile.json" --no-prefill-fp8)
  [[ -n ${TF_EXPERT_TENSOR_PARALLEL_SIZE:-} ]] || args+=(--expert-tensor-parallel-size 4)
  # Thinking stays on (the server default); TF_THINKING=off adds --no-thinking. HTTP
  # host payload grows with each reply (prompt + paid reply tokens + decoded text), so
  # all 32 lanes admit thinking requests. The queue timeout lets requests wait for FIFO
  # and host payload capacity (and a growing reply for more) instead of an immediate 503.
  [[ ${TF_THINKING:-on} == on ]] || args+=(--no-thinking)
  args+=(--queue-timeout-s "${TF_QUEUE_TIMEOUT_S:-900}")
  # Host-RAM prefix tier: empty keeps the server default (32 GiB on this path), 0 turns it off.
  [[ -z ${TF_PREFIX_HOST_BYTES:-} ]] || args+=(--prefix-host-bytes "$TF_PREFIX_HOST_BYTES")
  export TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN=grouped-v1 TF_SMALL_GATHER_BYTES=262144
}
