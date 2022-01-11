


echo "this code is used to run LL model with diff arch"


## declare an array variable
declare -a dataset_list=(  "DBLP" )
declare -a decoder_list=( "multi_inner_product")
declare -a Layers=(  4 )
for i in "${dataset_list[@]}"
do
        for k in "${decoder_list[@]}"
                do
                        ## now loop through the above array
                        for j in ${Layers[@]}
                        do
                           echo "$i"
                           echo "$j"
                           echo "$k"
                           # or do whatever with individual element of the array
                           nohup python -u VGAE_FrameWork.2.1.py  -dataset "$i" -decoder_type "$k" -NofRels $j >  res/"${i} ${k} ${j} HeteroKIARGCN"
                        done
                done
done
