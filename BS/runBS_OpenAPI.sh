#!/bin/bash
for S in `seq 0 8`
do
cd BS${S}; ./runBS${S}.sh -m DPCPP -p 4 -d 0  >& resultsBS${S}_DPCPP_INTEL_ARC770.out; cd ..
done
