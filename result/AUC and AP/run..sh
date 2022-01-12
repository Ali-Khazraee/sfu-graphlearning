


echo "this code is used to run LL model with diff arch"


## declare an array variable
declare -a dataset_list=( "cora" "citeseer" "ACM" "IMDB"  "pubmed" "DBLP")
declare -a decoder_list=("InnerDot")
declare -a encoder_list=("mixture_of_NGCNs")
declare -a Layers=(1 4  6 8  )
for i in "${dataset_list[@]}"
do	
	for en in "${encoder_list[@]}"
	do
        	for k in "${decoder_list[@]}"
                do
                        ## now loop through the above array
                        for j in ${Layers[@]}
                        do
                           	echo "$i"
                           	echo "$j"
                           	echo "$k"
				echo "$en"
                           	# or do whatever with individual element of the array
                           	nohup python -u VGAE_FrameWork.2.1.py  -dataset "$i" -decoder_type "$k" -encoder_type "$en" -NofRels $j >  res/"${i} ${en} ${k} ${j}"
                        done
                done
	done
done
