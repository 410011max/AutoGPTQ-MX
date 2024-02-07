# get results from python, then push to array in bash
model_name="meta-llama/Llama-2-13b-hf"
n_ctx_arr=(512)
mx_format_arr=(int8 fp6_e3m2 fp6_e2m3 fp4 int4)

# 2d ppl results array
declare -A ppl_arr

#function to get ppl from python get_ppl (model_name, n_ctx, mx_format)
function get_ppl {
    local model_name=$1
    local n_ctx=$2
    local mx_format=$3
    local ppl=$(python examples/benchmark/perplexity.py --model_name $model_name --n_ctx $n_ctx --n_batch $n_ctx --mx --mx_format $mx_format --no_tqdm 2>&1)
    echo $ppl | awk '{print $NF}'
}

function get_org_ppl {
    local model_name=$1
    local n_ctx=$2
    local ppl=$(python examples/benchmark/perplexity.py --model_name $model_name --n_ctx $n_ctx --n_batch $n_ctx --no_tqdm 2>&1)
    echo $ppl | awk '{print $NF}'
}

# function to get elapsed time, hour, minutes and seconds
function elapsed_time {
    local SECONDS=$1
    local minutes=$((SECONDS / 60))
    local seconds=$((SECONDS % 60))
    echo "$minutes:$seconds"
}

# get ppl from python and push to array
for n_ctx in ${n_ctx_arr[@]}
do
    SECONDS=0
    # echo "Running n_ctx: $n_ctx, original fp16..."
    ppl_arr[$n_ctx,fp16]=$(get_org_ppl $model_name $n_ctx)
    echo "ppl_arr[$n_ctx,fp16]: ${ppl_arr[$n_ctx,fp16]} --- Time elapsed: $(elapsed_time $SECONDS)"

    for mx_format in ${mx_format_arr[@]}
    do
        SECONDS=0
        # echo "Running n_ctx: $n_ctx, mx_format: $mx_format..."
        ppl_arr[$n_ctx,$mx_format]=$(get_ppl $model_name $n_ctx $mx_format)
        echo "ppl_arr[$n_ctx,$mx_format]: ${ppl_arr[$n_ctx,$mx_format]} --- Time elapsed: $(elapsed_time $SECONDS)"
    done
done

# print ppl array in table format
echo -e "\n\n"
echo -e "model_name: $model_name"
echo -e "n_ctx\t|int8\t|fp6_e3m2\t|fp6_e2m3\t|fp4\t|int4"
echo -e "----------------------------------------"
for n_ctx in ${n_ctx_arr[@]}
do
    echo -e "$n_ctx\t|${ppl_arr[$n_ctx,fp16]}\t|${ppl_arr[$n_ctx,int8]}\t|${ppl_arr[$n_ctx,fp6_e3m2]}\t\t|${ppl_arr[$n_ctx,fp6_e2m3]}\t\t|${ppl_arr[$n_ctx,fp4]}\t|${ppl_arr[$n_ctx,int4]}"
done
echo -e "\n\n"
exit 0