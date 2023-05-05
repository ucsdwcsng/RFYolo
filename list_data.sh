#!/bin/bash

cd $1

# echo $1;
# find . -name "*.png" > ./all.txt
# find . -name "*.s.png" > ./all.s.txt
# find . -name "*.a.png" > ./all.a.txt
# find . -name "*.na.png" > ./all.na.txt
# find . -name "*.ca.png" > ./all.ca.txt
# find . -name "*.nca.png" > ./all.nca.txt

# for d in $(ls -df */)
# do 
#     echo $d;
#     find $d -name "*.png" > $d/all.txt
#     find $d -name "*.s.png" > $d/all.s.txt
#     find $d -name "*.a.png" > $d/all.a.txt
#     find $d -name "*.na.png" > $d/all.na.txt
#     find $d -name "*.ca.png" > $d/all.ca.txt
#     find $d -name "*.nca.png" > $d/all.nca.txt
# done


echo $1;

ks=(1 4 16 64 128)

find . -name "*.png" > ./all.txt
for k in ${ks[@]}
do 
    find . -name "*.k${k}.png" > ./all.k${k}.txt
done


for d in $(ls -df */)
do 
    echo $d;
    find $d -name "*.png" > $d/all.txt
    for k in ${ks[@]}
    do 
        find . -name "*.k${k}.png" > $d/all.k${k}.txt
    done
done

