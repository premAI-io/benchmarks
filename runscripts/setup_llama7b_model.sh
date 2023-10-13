if (which git >/dev/null); then
  # fix this to work on linux as well
  # use packagecloud: https://packagecloud.io/github/git-lfs
  (git lfs &>/dev/null) || ( \
    brew install git-lfs; \
    echo "Installed git-lfs")
  git lfs install
  # https is not supported any longer
  git clone git@hf.co:meta-llama/Llama-2-7b /tmp/llama-2-7b
else
  echo "git not found!"
fi
