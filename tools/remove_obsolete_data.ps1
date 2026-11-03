$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dataRoot = (Resolve-Path -LiteralPath (Join-Path $project 'data')).Path
$provenance = Join-Path $dataRoot 'raw\sentinel_history_2021_2025\provenance'
New-Item -ItemType Directory -Path $provenance -Force | Out-Null
$sourceGeometry = Join-Path $dataRoot 'raw\maroc\oliveraies_reference.geojson'
$targetGeometry = Join-Path $provenance 'oliveraies_reference.geojson'
Copy-Item -LiteralPath $sourceGeometry -Destination $targetGeometry -Force
if ((Get-FileHash -LiteralPath $sourceGeometry).Hash -ne (Get-FileHash -LiteralPath $targetGeometry).Hash) { throw 'Geometry copy mismatch' }
foreach ($name in @('morocco_reference_figshare_v2.json','morocco_reference_manifest.json','morocco_reference_audit.json')) {
    Copy-Item -LiteralPath (Join-Path $dataRoot "metadata\$name") -Destination (Join-Path $provenance $name) -Force
}
$targets = @()
$targets += Get-ChildItem -LiteralPath (Join-Path $dataRoot 'raw') -Force | Where-Object { $_.Name -ne 'sentinel_history_2021_2025' }
$targets += Get-ChildItem -LiteralPath (Join-Path $dataRoot 'processed') -Force | Where-Object { $_.Name -notin @('sentinel_history_2021_2025','model_inputs_sentinel') }
$targets += Get-Item -LiteralPath (Join-Path $dataRoot 'metadata')
# Resolve every target before any recursive deletion; allow only children of this project's data directory.
$resolved = @($targets | ForEach-Object { (Resolve-Path -LiteralPath $_.FullName).Path })
foreach ($path in $resolved) {
    if (-not $path.StartsWith($dataRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe target: $path" }
    if ($path -eq (Join-Path $dataRoot 'raw\sentinel_history_2021_2025') -or $path -eq (Join-Path $dataRoot 'processed\model_inputs_sentinel')) { throw 'Protected target' }
}
$audit = Join-Path $project 'outputs\sentinel_history_2021_2025\removed_obsolete_data.json'
$resolved | ConvertTo-Json | Set-Content -LiteralPath $audit -Encoding utf8
foreach ($path in $resolved) {
    Remove-Item -LiteralPath $path -Recurse -Force
}
Write-Output "Removed $($resolved.Count) obsolete data entries. New satellite data and model inputs retained."
