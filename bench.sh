# pick platform, we can avoid this by automating it(long term goal)
case $1 in
  "mac")
    VENDOR="apple"
    OS="mac"
    PLATFORM=`uname -s`
    ARCH=`uname -m`
    if ! [ "$PLATFORM" = "Darwin" ] ; then
      echo "OS $OS incompatible with PLATFORM $PLATFORM"
      exit -1
    fi
  ;;
  "linux")
    VENDOR="unknown"
    OS="linux"
    PLATFORM=`uname -s`
    ARCH=`uname -m`
    if ! [ $PLATFORM = "Linux" ]; then
      echo "OS $OS incompatible with PLATFORM $PLATFORM"
      exit -1
    fi
  ;;
  "windows")
    echo "unsupported"
    exit -1
  ;;
  *)
    echo "$1"
    echo "only [mac, linux, windows] options allowed!"
    exit -1
  ;;
esac

# pick device
case $2 in
  "cpu")
    DEVICE="CPU";;
  "gpu")
    DEVICE="CPU";;
  *)
    echo "$2"
    echo "only [cpu, gpu] options allowed!"
    exit -1;;
esac

echo "==================="
echo "Running benchmarks for,"
echo ""
echo "OS:        $OS"
echo "PLATFORM:  $PLATFORM"
echo "ARCH:      $ARCH"
echo "VENDOR:    $VENDOR"
echo "DEVICE:    $DEVICE"
echo "==================="

## ensure max_tokens gen == 100

echo "Benching tinygrad with llama 7B dynamically quantized"
echo "DEVICE $DEVICE"

if ! [ $DEVICE = "GPU" ]; then
  source ./src/setup/tinygrad.sh
  ## TODO(swarnim): run the below in a loop with different prompts and store the perf metrics
  CPU=1 ./src/run/tinygrad.sh "write a note about distributed machine learning"
  deactivate
else
  source ./src/setup/tinygrad.sh
  ## defaults to METAL/GPU
  ## TODO(swarnim): run the below in a loop with different prompts and store the perf metrics
  ./src/run/tinygrad.sh "write a note about distributed machine learning"
  deactivate
fi


echo "Benching llama.cpp with llama 7B quantized model"
echo "DEVICE $DEVICE"

if ! [ $DEVICE = "GPU" ]; then
  source ./src/setup/llama.cpp.sh
  ## TODO(swarnim): run the below in a loop with different prompts and store the perf metrics
  ./src/run/llama.cpp.sh "write a note about distributed machine learning" -ngl 0
  deactivate
else
  source ./src/setup/llama.cpp.sh
  ## defaults to METAL/GPU
  ## TODO(swarnim): run the below in a loop with different prompts and store the perf metrics
  ./src/run/llama.cpp.sh "write a note about distributed machine learning"
  deactivate
fi


