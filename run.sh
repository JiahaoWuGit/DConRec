#=============================================================================gowalla-merged
# for model in 'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset gowalla-merged --test_model_selection ${model}
# done
#finish all

# for model in 'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset gowalla-merged --model_selection ${model}
# done
#finish all

# for model in 'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset gowalla-merged --model_selection ${model} --test_model_selection 'NGCF'
# done
#finish all

# for model in 'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset gowalla-merged --model_selection ${model} --test_model_selection 'LightGCN'
# done
#finish all

# for model in 'LightGCN' 'NGCF' #'XSimGCL'
# do
#     python main.py --dataset gowalla-merged --model_selection ${model} --test_model_selection 'XSimGCL'
# done
#finish all

#=============================================================================dianping
# for model in  'LightGCN' 'NGCF' #'BPRMF' 'XSimGCL'
# do
#     python main.py --dataset dianping --model_selection ${model} --test_model_selection 'BPRMF'
# done
# #‘BPRMF' 'XSimGCL', finish

# for model in 'BPRMF' 'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset dianping --model_selection ${model} --test_model_selection 'NGCF'
# done

# for model in 'LightGCN' 'NGCF' #'BPRMF' 'XSimGCL' 
# do
#     python main.py --dataset dianping --model_selection ${model} --test_model_selection 'LightGCN'
# done
# #‘BPRMF' 'XSimGCL', finish


# for model in  'BPRMF'  'XSimGCL' 'LightGCN' 'NGCF'
# do
#     python main.py --dataset dianping --model_selection ${model} --test_model_selection 'XSimGCL'
# done

#=============================================================================ml-20m
# for model in   #'BPRMF' 'LightGCN''NGCF' 'XSimGCL'
# do
#     python main.py --dataset ml-20m --model_selection ${model} --test_model_selection 'BPRMF'
# done
# #BPRMF, LightGCN, NGCF, XSimGCL, finish

for model in 'BPRMF' 'XSimGCL' 'LightGCN' 'NGCF'
do
    python main.py --dataset ml-20m --model_selection ${model} --test_model_selection 'NGCF'
done

for model in 'BPRMF' 'XSimGCL' 'LightGCN' 'NGCF'
do
    python main.py --dataset ml-20m --model_selection ${model} --test_model_selection 'LightGCN'
done

for model in   'LightGCN' 'NGCF' #'XSimGCL'
do
    python main.py --dataset ml-20m --model_selection ${model} --test_model_selection 'XSimGCL'
done
