Get-Process | Where-Object { $_.ProcessName -match 'node|tsx|python|qdrant' } |
  Sort-Object CPU -Descending |
  Select-Object -First 12 Id, ProcessName, @{n='CPU_s';e={[math]::Round($_.CPU,0)}}, @{n='RSS_MB';e={[math]::Round($_.WorkingSet64/1MB,0)}} |
  Format-Table -AutoSize
Write-Output ("CPU cores: " + [Environment]::ProcessorCount)
Write-Output ("Load %: " + (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average)
