#!/bin/bash

function HELP {
  echo "Usage: ./runBS.sh -m MODE"
  exit 1
}

#parse options
while getopts ":m:h:p:d:e" FLAG; do
  case "$FLAG" in
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

# Build the code
# make -j `nproc`

if [ -z $mode ]; then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

for bs in `seq 0 8`
do
./BS${bs}/BS${bs} -m $mode -pl $plat -d $devi
done

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
