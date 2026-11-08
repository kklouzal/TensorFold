#!/bin/bash
# Personal deployment ceilings only. The engine still proves actual capacity.
# Sourcing defines one adapter; call it before constructing serving/benchmark argv.

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
    rotorquant-iso4-sm70-v1) ceiling=32 ;;
    *) return 0 ;;  # No invented ceiling/default for another codec/profile.
  esac
  _tf_v100_ceiling TF_PARALLEL "$ceiling" || return
  _tf_v100_ceiling TF_KV_FULL_WINDOWS "$ceiling" || return
  effective=$TF_PARALLEL
  (( TF_KV_FULL_WINDOWS >= effective )) || effective=$TF_KV_FULL_WINDOWS
  pending=${TF_MAX_PENDING_REQUESTS:-$effective}
  _tf_v100_positive TF_MAX_PENDING_REQUESTS "$pending" || return
  # Reuse the exact decimal ceiling validation; P may exceed a lower GPU
  # ceiling for bounded waiting, but never this personal profile's ceiling.
  TF_MAX_PENDING_REQUESTS=$pending
  _tf_v100_ceiling TF_MAX_PENDING_REQUESTS "$ceiling" || return
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
