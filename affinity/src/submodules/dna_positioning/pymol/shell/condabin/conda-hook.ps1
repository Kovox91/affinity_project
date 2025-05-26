$Env:CONDA_EXE = "/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda"
$Env:_CONDA_EXE = "/home/appveyor/projects/pymol-incentive/ci-bundle-311/_pymolconda/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs