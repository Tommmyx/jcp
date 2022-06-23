#/bin/bash
mkdir -p lib
# libsvm
g++ -O2 -Wall -g -fopenmp -o lib/libsvm-jni.so jni/libsvm-jni.cpp -shared -fpic -Ijni/include -I$JAVA_HOME/include/ -I$JAVA_HOME/include/linux/ -I../libsvm.git/ -L./lib -lsvm
