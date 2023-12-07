#!/bin/bash

mpi="mpirun -np 1 "
exe=./BS

function HELP {
    echo "Usage: ./run.sh -m MODE -e ELEMENT -n NDOFS -P problem -p degree -b bytes [-s] [-t] [-v]"
    exit 1
}

function RUN {
    if [ "$element" == "Hex" ] || [ "$element" == "Tet" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/($2*$2*$2))^(1/3)+0.499 }')
        $mpi $exe -m $mode -P $problem -e $element -nx $N -ny $N -nz $N -p $p
    elif [ "$element" == "Quad" ] || [ "$element" == "Tri" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/($2*$2))^(1/2)+0.499 }')
        $mpi $exe -m $mode -P $problem -e $element -nx $N -ny $N -p $p
    fi
}

function CHECKDEGREE {
    if [ "$element" = "Tet" ]; then
        pmax=9
    else
        pmax=15
    fi
}

#defaults
element=Hex
ndofs=4000000
bytes=1073741824
sweep=false
tune=false
verbose=false
problem=-1
p=3

#parse options
while getopts :m:e:n:P:p:b:sktvh FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    e)
        element=$OPTARG
        [[ ! $element =~ Tri|Tet|Quad|Hex ]] && {
            echo "Incorrect element type provided"
            exit 1
        }
        ;;
    n)
        ndofs=$OPTARG
        ;;
    P)
        problem=$OPTARG
        ;;
    p)
        p=$OPTARG
        ;;
    b)
        b=$OPTARG
        ;;
    s)
        sweep=true
        ;;
    t)
        tune=true
        ;;
    v)
        verbose=true
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

if [ "$problem" = 0 ] || [ "$problem" = 1 ] || [ "$problem" = 2 ] || [ "$problem" = 3 ] || [ "$problem" = 4 ] || [ "$problem" = 5 ]; then
    if [ "$sweep" = true ] ; then
        exe+=" -sw "
    fi
fi
if [ "$tune" = true ] ; then
    exe+=" -t "
fi
if [ "$verbose" = true ] ; then
    exe+=" -v "
fi

if [ "$problem" = -1 ] ; then
    #Run all benchmarks
    for problem in $(seq 0 8)
    do
        RUN
    done
else
    RUN
fi

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
