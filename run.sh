#!/bin/bash

function HELP {
  echo "Usage: ./run.sh -m MODE"
  exit 1
}

#parse options
while getopts :m:p:d:h FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial|DPCPP ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    p)
        plat=$OPTARG;
        echo "platform=" $plat;;
    d)
        devi=$OPTARG;
        echo "device=" $devi;;
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

cd BS; ./runBS.sh -m $mode -p $plat -d $devi; cd ..

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
