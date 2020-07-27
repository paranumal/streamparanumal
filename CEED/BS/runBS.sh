#!/bin/bash

function HELP {
  echo "Usage: ./runBS.sh -m MODE"
  exit 1
}

#parse options
while getopts :m:h FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    h)  #show help
        HELP
        ;;
    \?) #unrecognized option - show help
        HELP
        ;;
  esac
done

if [ -z $mode ]
then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

# Build the code

if [ ! -d "../../occa" ]; then
  cd ../../
  git clone https://github.com/libocca/occa
  cd occa; make -j `nproc`
  cd ../CEED/BS
fi

export OCCA_DIR=${PWD}/../../occa

make -j `nproc`

cd BS1; ./runBS1.sh -m $mode > ../results/BS1MI60Corona.out; cd ..; sleep 30
cd BS2; ./runBS2.sh -m $mode > ../results/BS2MI60Corona.out; cd ..; sleep 30
cd BS3; ./runBS3.sh -m $mode > ../results/BS3MI60Corona.out; cd ..; sleep 30
cd BS4; ./runBS4.sh -m $mode > ../results/BS4MI60Corona.out; cd ..; sleep 30
cd BS5; ./runBS5.sh -m $mode > ../results/BS5MI60Corona.out; cd ..; sleep 30
cd BS6; ./runBS6.sh -m $mode > ../results/BS6MI60Corona.out; cd ..; sleep 30
cd BS7; ./runBS7.sh -m $mode > ../results/BS7MI60Corona.out; cd ..; sleep 30
cd BS8; ./runBS8.sh -m $mode > ../results/BS8MI60Corona.out; cd ..; sleep 30

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
