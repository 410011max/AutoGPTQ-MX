# get results from python, then push to array in bash
model_name="meta-llama/Llama-2-13b-hf"
mx_format_arr=(int8 fp6_e3m2 fp4 int4)
mx_block_size_arr=(32)

# 2d ppl results array
declare -A ppl_arr

# function to get elapsed time, hour, minutes and seconds
function elapsed_time {
    local SECONDS=$1
    local minutes=$((SECONDS / 60))
    local seconds=$((SECONDS % 60))
    echo "$minutes:$seconds"
}

#function to get ppl from python get_ppl (model_name, mx_format, mx_block_size)
function get_ppl {
    local model_name=$1
    local mx_format=$2
    local mx_block_size=$3
    local ppl=$(python examples/benchmark/perplexity.py --model_name $model_name --mx --mx_format $mx_format --mx_block_size $mx_block_size --no_tqdm 2>&1)
    echo $ppl | awk '{print $NF}'
}

# get ppl from python and push to array
for mx_format in ${mx_format_arr[@]}
do
    SECONDS=0
    for mx_bs in ${mx_block_size_arr[@]}
    do
        SECONDS=0
        ppl_arr[$mx_format,$mx_bs]=$(get_ppl $model_name $mx_format $mx_bs)
        echo "ppl_arr[$mx_format,$mx_bs]: ${ppl_arr[$mx_format,$mx_bs]} --- Time elapsed: $(elapsed_time $SECONDS)"
    done
done

# print ppl array in table format
echo -e "\n\n"
echo -e "model_name: $model_name"
echo -e "mx_format\t|int8\t|fp6_e3m2\t|fp4\t|int4"
echo -e "----------------------------------------"
for mx_format in ${mx_format_arr[@]}
do
    echo -e "$mx_format\t|${ppl_arr[$mx_format,8]}\t|${ppl_arr[$mx_format,16]}\t|${ppl_arr[$mx_format,32]}\t|${ppl_arr[$mx_format,64]}\t|${ppl_arr[$mx_format,128]}"
done
echo -e "\n\n"
exit 0



