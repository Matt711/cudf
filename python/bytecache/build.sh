#!/usr/bin/env bash
# Build the bytecache extension against the tpch conda env (libcudf, kvikio, pylibcudf from the env). Output: _bytecache*.so next to this script.
set -euo pipefail
E=${ENV_PREFIX:-/home/ubuntu/mmurray2/envs/tpch}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$E/bin/python
SP=$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
PYINC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
EXT=.abi3.so   # pylibcudf is built against the limited API (Cython 3.2.9 "limited"); a subclass of its Datasource must match
CXX=$E/bin/x86_64-conda-linux-gnu-g++
cd "$HERE"
PYTHONNOUSERSITE=1 $E/bin/cython --cplus -3 -I "$SP" _bytecache.pyx -o _bytecache.cpp
$CXX -O2 -g -fPIC -shared -std=c++20 -Wall -Wno-unused-function \
  -I"$HERE" -I"$E/include" -I"$E/targets/x86_64-linux/include" -I"$E/include/rapids" -DCYTHON_LIMITED_API=1 -DPy_LIMITED_API=0x030A0000 -DKVIKIO_CUFILE_FOUND -DKVIKIO_CUFILE_VERSION_API_FOUND -DKVIKIO_LIBCURL_FOUND -I"$PYINC" -I"$SP" \
  _bytecache.cpp bytecache.cpp -o "_bytecache$EXT" \
  -L"$E/lib" -Wl,-rpath,"$E/lib" -lcudf -lkvikio -lcudart -lcurl
echo "built _bytecache$EXT"
