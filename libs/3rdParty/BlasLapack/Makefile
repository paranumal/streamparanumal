#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#####################################################################################

ifndef CEED_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
  $(error cannot locate ${PWD}/../../make.top)
else
  include ../../make.top
endif
endif

FC = gfortran
AR = ar

SDIR = ./
INCLUDES = -I.
OPTFLAGS = -O3
ARFLAGS  = -cr

SRC = $(wildcard $(SDIR)/*.f)
OBJS = $(subst $(SDIR)/,$(SDIR),$(SRC:.f=.o))

FCFLAGS = $(FCOPTIONS) $(OPTFLAGS)

.PHONY: all lib clean

.SUFFIXES: .f

all: lib

lib: libBlasLapack.a silentUpdate

libBlasLapack.a: $(OBJS)
ifneq (,${verbose})
	$(AR) $(ARFLAGS) libBlasLapack.a *.o
else
	@printf "%b" "$(LIB_COLOR)Building $(@F)$(NO_COLOR)\n";
	@$(AR) $(ARFLAGS) libBlasLapack.a *.o
endif

silentUpdate:
	@true

.f.o:
ifneq (,${verbose})
	$(FC) $(FCFLAGS) -c $*.f
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(FC) $(FCFLAGS) -c $*.f
endif

clean:
	rm -f libBlasLapack *.o
