#!/usr/bin/env bash
# Get a freshly rented box ready while the model is still converting elsewhere: system
# packages, the venv and the pinned llama.cpp build (~10 min of every node's start). An
# extra imatrix or quant box then begins its real work the minute the BF16 is on the hub.
#
# env: STEM, LLAMA_COMMIT
# out: nothing on the hub; /opt/venv and the llama.cpp build on this box
. "${PIPE:-/opt/pipeline}/lib/node_common.sh"
ensure_llama
done_node "\"gpus\": $NGPU"
