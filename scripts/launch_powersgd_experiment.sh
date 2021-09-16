#!/bin/bash
#
# Different compression schemes can be enabled by exporting environment varialbe `COMM_HOOK_TYPE` as one of the following values:
# 1) FP16_COMPRESS
# 2) POWER_SGD
# 3) FP16_POWER_SGD
# 4) BATCHED_POWER_SGD
# 5) FP16_BATCHED_POWER_SGD

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
train.py \
--task masked_lm \
--tokens-per-sample 16 \
--batch-size 8 \
--update-freq 1 \
--arch model_parallel_roberta \
--encoder-embed-dim 2560 \
--encoder-ffn-embed-dim 10240 \
--encoder-layers 12 \
--encoder-attention-heads 32 \
--dropout 0.1 \
--attention-dropout 0.1 \
--activation-dropout 0.0 \
--optimizer adam \
--weight-decay 0.01 \
--lr 0.001 \
--log-format simple \
--log-interval 1 \
--max-epoch 3 \
--no-save \
--model-parallel-size 1
--skip-invalid-size-inputs-valid-test \
$WIKITEXT_DATA_PATH
