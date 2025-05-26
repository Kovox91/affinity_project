export MAMBA_ROOT_PREFIX="/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda"
__mamba_setup="$("/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda/bin/mamba" shell hook --shell posix 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda/bin/mamba"  # Fallback on help from mamba activate
fi
unset __mamba_setup
