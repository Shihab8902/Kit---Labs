# Kit env — keeps everything on D:
$kit = "D:\Web Projects\Kit"
New-Item -ItemType Directory -Force -Path "$kit\.cache\pip", "$kit\.cache\hf", "$kit\.cache\torch", "$kit\.cache\matplotlib", "$kit\.tmp" | Out-Null
$env:HF_HOME = "$kit\.cache\hf"
$env:HF_DATASETS_CACHE = "$kit\.cache\hf\datasets"
$env:TORCH_HOME = "$kit\.cache\torch"
$env:PIP_CACHE_DIR = "$kit\.cache\pip"
$env:MPLCONFIGDIR = "$kit\.cache\matplotlib"
$env:TMP = "$kit\.tmp"
$env:TEMP = "$kit\.tmp"
$env:PYTHONPATH = "$kit"
Write-Host "Kit env set. Caches and temp point to D:."
