param([ValidateSet('serial','openmp','cuda','hybrid')]$Mode='serial',[int]$Size=256,[int]$Threads=4)
$ErrorActionPreference = 'Stop'
& '.\build\denoise.exe' denoise --mode $Mode --width $Size --height $Size --threads $Threads --clean-output 'results\clean.pgm' --noisy-output 'results\noisy.pgm' --output "results\$Mode.pgm"
