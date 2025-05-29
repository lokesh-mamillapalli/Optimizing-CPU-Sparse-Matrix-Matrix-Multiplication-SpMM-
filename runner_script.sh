#!/bin/bash
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build
cmake .. && make -j
cd ..
./build/bin/tester 2048 2048 2048 111
