#!/bin/bash
# script to perform multiple testing.
base_path="./configs/$1/$1_"
base_path_generation="./configs/$1_generation/$1_generation_"
base_path_log="./models/$1/"

for i in $(seq -w 1 $2);
do
    cfg_path_generation="${base_path_generation}${i}.yaml"
    cfg_path="${base_path}${i}.yaml"
    path_log="${base_path_log}${i}/$1_${i}.txt"
    if [ $i -le $3 ];
    then
        case $4 in
            -g)
                echo "Using file: ${cfg_path_generation} to generate data.";
                python3 kcurves/generate_synthetic.py $cfg_path_generation;
            ;;
            -t)
                echo "Using file: ${cfg_path} to perform training and test.";
                echo ${path_log};
                python3 kcurves train $cfg_path;
                python3 kcurves test $cfg_path;
            ;;
            -m)
                if [ $1 = "synthetic_clusters" ] || [ $1 = "synthetic_functions" ] || [ $1 = "synthetic_lines" ];
                then
                  echo "Using file: ${cfg_path_generation} to generate data.";
                  python3 kcurves/generate_synthetic.py $cfg_path_generation;
                fi
                echo "Using file: ${cfg_path} to perform multiple testing.";
                echo ${path_log};
                python3 kcurves multiple_testing $cfg_path;
            ;;
            -a)
                echo "Using file: ${cfg_path_generation} to generate data.";
                python3 kcurves/generate_synthetic.py $cfg_path_generation;
                echo "Using file: ${cfg_path} to perform training and test.";
                python3 kcurves train $cfg_path;
                python3 kcurves test $cfg_path;
            ;;
        esac
        echo "------------------------------------------------------";
    fi
done