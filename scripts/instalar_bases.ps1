# instalar_bases.ps1 — con TLS forzado y errores visibles
# Ejecutar como Administrador

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$mysql  = 'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe'
$tmpDir = 'C:\Users\DAVID\Desktop\agent_skills\scripts\mysql_downloads'
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

# ── SAKILA ────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Descargando Sakila ===" -ForegroundColor Cyan
$sakilaZip = "$tmpDir\sakila.zip"

try {
    $wc = New-Object System.Net.WebClient
    Write-Host "  Conectando a downloads.mysql.com..." -ForegroundColor Yellow
    $wc.DownloadFile("https://downloads.mysql.com/docs/sakila-db.zip", $sakilaZip)
    $size = (Get-Item $sakilaZip).Length / 1KB
    Write-Host "  Descargado: $([math]::Round($size,0)) KB" -ForegroundColor Green
} catch {
    Write-Host "  ERROR descargando: $_" -ForegroundColor Red
    Write-Host "  Intentando con curl..." -ForegroundColor Yellow
    curl.exe -L -o $sakilaZip "https://downloads.mysql.com/docs/sakila-db.zip"
}

if (!(Test-Path $sakilaZip) -or (Get-Item $sakilaZip).Length -lt 1000) {
    Write-Host "  FALLO la descarga de Sakila." -ForegroundColor Red
} else {
    Write-Host "  Descomprimiendo..." -ForegroundColor Yellow
    Expand-Archive -Path $sakilaZip -DestinationPath $tmpDir -Force

    $schemaFile = "$tmpDir\sakila-db\sakila-schema.sql"
    $dataFile   = "$tmpDir\sakila-db\sakila-data.sql"

    Write-Host "  Instalando schema..." -ForegroundColor Yellow
    $out = cmd /c "`"$mysql`" -u root -p123 < `"$schemaFile`"" 2>&1
    if ($out) { Write-Host "  $out" -ForegroundColor $(if ($out -match 'ERROR') {'Red'} else {'Gray'}) }

    Write-Host "  Instalando datos..." -ForegroundColor Yellow
    $out = cmd /c "`"$mysql`" -u root -p123 < `"$dataFile`"" 2>&1
    if ($out) { Write-Host "  $out" -ForegroundColor $(if ($out -match 'ERROR') {'Red'} else {'Gray'}) }

    Write-Host "  Sakila lista." -ForegroundColor Green
}

# ── WORLD ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Descargando World ===" -ForegroundColor Cyan
$worldZip = "$tmpDir\world.zip"

try {
    $wc = New-Object System.Net.WebClient
    Write-Host "  Conectando..." -ForegroundColor Yellow
    $wc.DownloadFile("https://downloads.mysql.com/docs/world-db.zip", $worldZip)
    $size = (Get-Item $worldZip).Length / 1KB
    Write-Host "  Descargado: $([math]::Round($size,0)) KB" -ForegroundColor Green
} catch {
    Write-Host "  ERROR descargando: $_" -ForegroundColor Red
    curl.exe -L -o $worldZip "https://downloads.mysql.com/docs/world-db.zip"
}

if (!(Test-Path $worldZip) -or (Get-Item $worldZip).Length -lt 1000) {
    Write-Host "  FALLO la descarga de World." -ForegroundColor Red
} else {
    Write-Host "  Descomprimiendo..." -ForegroundColor Yellow
    Expand-Archive -Path $worldZip -DestinationPath $tmpDir -Force

    $worldFile = "$tmpDir\world-db\world.sql"

    Write-Host "  Instalando..." -ForegroundColor Yellow
    $out = cmd /c "`"$mysql`" -u root -p123 < `"$worldFile`"" 2>&1
    if ($out) { Write-Host "  $out" -ForegroundColor $(if ($out -match 'ERROR') {'Red'} else {'Gray'}) }

    Write-Host "  World lista." -ForegroundColor Green
}

# ── VERIFICACION ──────────────────────────────────────────────
Write-Host ""
Write-Host "=== RESULTADO FINAL ===" -ForegroundColor Cyan
cmd /c "`"$mysql`" -u root -p123 -e `"SHOW DATABASES;`"" 2>&1 |
    Where-Object { $_ -notmatch 'Warning' }
