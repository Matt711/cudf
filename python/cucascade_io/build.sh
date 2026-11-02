#!/usr/bin/env bash
# Build the cuCascade datasource glue against the tpch env (libcudf, pylibcudf) and the cuCascade build in cache/cucascade-build.
set -euo pipefail
E=${ENV_PREFIX:-/home/ubuntu/mmurray2/envs/tpch}
CC_SRC=${CC_SRC:-/home/ubuntu/mmurray2/repos/cucascade}
CC_BLD=${CC_BLD:-/home/ubuntu/mmurray2/cache/cucascade-build}
DEPS=${DEPS:-/home/ubuntu/mmurray2/envs/cc-deps}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$E/bin/python
SP=$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
PYINC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
CXX=$E/bin/x86_64-conda-linux-gnu-g++
cd "$HERE"
PYTHONNOUSERSITE=1 $E/bin/cython --cplus -3 -I "$SP" _cucascade_io.pyx -o _cucascade_io.cpp
$CXX -O2 -g -fPIC -shared -std=c++20 -Wall -Wno-unused-function \
  -I"$HERE" -I"$CC_SRC/include" -I"$E/include" -I"$E/targets/x86_64-linux/include" -I"$E/include/rapids" -I"$DEPS/include" -I"$CC_BLD/_deps/concurrentqueue-src" \
  -DCUCASCADE_HAS_KVIKIO -DCUCASCADE_MOODYCAMEL_NAMESPACE=moodycamel -DBS_THREAD_POOL_ENABLE_PAUSE=1 \
  -DCCCL_DISABLE_PDL -DCUB_DISABLE_NAMESPACE_MAGIC -DCUB_IGNORE_NAMESPACE_MAGIC_ERROR -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR \
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DKVIKIO_CUFILE_FOUND -DKVIKIO_CUFILE_VERSION_API_FOUND -DKVIKIO_LIBCURL_FOUND \
  -DCYTHON_LIMITED_API=1 -DPy_LIMITED_API=0x030A0000 -I"$PYINC" -I"$SP" \
  _cucascade_io.cpp cc_glue.cpp -o _cucascade_io.abi3.so \
  -L"$CC_BLD" -lcucascade_cudf -lcucascade_io -lcucascade -L"$E/lib" -L"$DEPS/lib" -lcudf -lrmm -lcudart \
  -Wl,-rpath,"$CC_BLD" -Wl,-rpath,"$E/lib" -Wl,-rpath,"$DEPS/lib"
echo "built _cucascade_io.abi3.so"
