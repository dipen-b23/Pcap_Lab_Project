param([ValidateSet('serial','openmp')]$Mode='serial',[int]$Size=512,[int]$Threads=4,[int]$Repetitions=5)
$ErrorActionPreference = 'Stop'
& '.\build\denoise.exe' benchmark --mode $Mode --width $Size --height $Size --threads $Threads --repetitions $Repetitions --csv "results\benchmark_$Mode.csv"
