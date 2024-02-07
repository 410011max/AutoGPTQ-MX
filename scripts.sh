export HF_HOME="/work/max410011/.cache/huggingface";

CUDA_VISIBLE_DEVICES=0 \
python examples/benchmark/perplexity.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--n_ctx 512 --n_batch 512

CUDA_VISIBLE_DEVICES=0 \
python examples/benchmark/perplexity.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--n_ctx 512 --n_batch 512 --mx --mx_format int8 --mx_block_size 32

CUDA_VISIBLE_DEVICES=0,1 \
python examples/benchmark/perplexity.py \
--model_name "meta-llama/Llama-2-13b-hf" \
--n_ctx 512 --n_batch 512 --mx --mx_format int8 --mx_block_size 32

CUDA_LAUNCH_BLOCKING=1
--model_name "Enoch/llama-7b-hf" \
--model_name "meta-llama/Llama-2-7b-hf" \


