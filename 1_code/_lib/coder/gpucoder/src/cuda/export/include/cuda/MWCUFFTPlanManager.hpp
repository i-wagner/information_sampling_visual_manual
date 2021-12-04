/* Copyright 2019 The MathWorks, Inc. */
#ifndef __MWCUFFTPLANMANGER_HPP__
#define __MWCUFFTPLANMANGER_HPP__

#include "cufft.h"

cufftHandle acquireCUFFTPlan(int nelem, cufftType type, int batch, int idist);

#endif
