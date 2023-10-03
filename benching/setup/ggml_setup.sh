## source this
## can also be executed!

# git clone --depth=1 https://github.com/ggerganov/ggml /tmp/ggml
# cd /tmp/ggml
# mkdir build && cd build
# cmake ..

# should we build everything?
# make -j4 gpt-2 gpt-j

# use model specific repo 
# whisper
# git clone https://github.com/ggerganov/whisper.cpp.git --depth=1 /tmp/whisper.cpp
# cd /tmp/whisper.cpp

# won't redownload if already downloaded
# make base.en
# make small.en
# make tiny.en
