#!/bin/bash
# script to save a test.
if [ ! -d "./tests/$1" ];
then
    mkdir "./tests/$1";
fi
if [ ! -d "./tests/$1/$2" ];
then
    mkdir "./tests/$1/$2";
fi

#mkdir "./tests/$1/$2/data";
mkdir "./tests/$1/$2/configs"
#mkdir "./tests/$1/$2/data/plots";
#mkdir "./tests/$1/$2/data/train";
#mkdir "./tests/$1/$2/data/test";
mkdir "./tests/$1/$2/models";
mkdir "./tests/$1/$2/evolution";
#mkdir "./tests/$1/$2/configs/generation";
mkdir "./tests/$1/$2/configs/train";


#cp -r "./data/$1/plots/"* "./tests/$1/$2/data/plots/";
#cp -r "./data/$1/train/"* "./tests/$1/$2/data/train/";
#cp -r "./data/$1/test/"*  "./tests/$1/$2/data/test/";
cp -r "./models/$1/"* "./tests/$1/$2/models/";
cp -r "./models/$1_evolution/"*  "./tests/$1/$2/evolution"
#cp -r "./configs/$1_generation/"* "./tests/$1/$2/configs/generation";
cp -r "./configs/$1/"* "./tests/$1/$2/configs/train";
