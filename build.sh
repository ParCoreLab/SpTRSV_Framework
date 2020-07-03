#!/usr/bin/env bash

# Copyright (c) 2020 Najeeb Ahmad
# All rights reserved.
#
# This file is part of SpTRSV framework. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

reportIfSuccessful() {
  if [ $? -eq 0 ]; then
    echo -e "\033[1;32mSUCCESS: $1.\033[m"
    return 0
  else
    echo -e "\033[1;31mFAILURE: $1.\033[m"
    exit 1
  fi
}

if [ "$1" = "clean" ]; then
	
	cd ./src/mkl_seq/
	make clean

	cd ../mkl_par/
	make clean

	cd ../cusparse_v1/
	make clean

	cd ../cusparse_v2_lvl_nolvl/
	make clean

	cd ../Benchmark_SpTRSM_using_CSC/SpTRSV_cuda
	make clean
	
	cd ../../matrix_feature_extractor/
	make clean
	exit 1
fi

echo "Building binaries ..."
echo "Building MKL sequential SpTRSV code ..."
cd ./src/mkl_seq/
make

reportIfSuccessful "Building MKL sequential SpTRSV code ..." 

echo "Building MKL parallel SpTRSV code ..."
cd ../mkl_par/
make

reportIfSuccessful "Building MKL parallel SpTRSV code ..."

echo "Building cuSPARSE v1 SpTRSV code ..."
cd ../cusparse_v1/
make

reportIfSuccessful "Building cuSPARSE v1 SpTRSV code ..."

echo "Building cuSPARSE v2 SpTRSV code ..."
cd ../cusparse_v2_lvl_nolvl/
make

reportIfSuccessful "Building cuSPARSE v2 SpTRSV code ..."

echo "Building SyncFree SpTRSV code ..."
cd ../Benchmark_SpTRSM_using_CSC/SpTRSV_cuda
make

reportIfSuccessful "Building SyncFree SpTRSV code ..."

echo "Building Feature Extraction Tool code ..."
cd ../../matrix_feature_extractor/
make

reportIfSuccessful "Building Feature Extraction Tool code ..."

echo "All SpTRSV binaries built successfully!"
