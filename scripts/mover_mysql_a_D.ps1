# =============================================================
# mover_mysql_a_D.ps1
# 1. Mueve el datadir de MySQL de C: a D:\MySQL\Data
# 2. Instala Sakila y World
# Ejecutar como Administrador en PowerShell
# =============================================================

$mysql    = 'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe'
$myini    = 'C:\ProgramData\MySQL\MySQL Server 8.0\my.ini'
$srcData  = 'C:\ProgramData\MySQL\MySQL Server 8.0\Data'
$destData = 'D:\MySQL\Data'
$tmpDir   = 'C:\Users\DAVID\Desktop\agent_skills\scripts\mysql_downloads'

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 1: Detener servicio MySQL80" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Stop-Service -Name MySQL80 -Force
Start-Sleep -Seconds 3
Write-Host "  MySQL detenido." -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 2: Crear carpeta D:\MySQL\Data" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $destData -Force | Out-Null
Write-Host "  Carpeta creada: $destData" -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 3: Copiar datos de C: a D: (puede tardar ~30s)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Copy-Item -Path "$srcData\*" -Destination $destData -Recurse -Force
Write-Host "  Copia completada." -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 4: Actualizar my.ini con nuevo datadir" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
$ini = Get-Content $myini -Raw
$ini = $ini -replace 'datadir=.*', "datadir=D:/MySQL/Data"
Set-Content -Path $myini -Value $ini -Encoding UTF8
Write-Host "  my.ini actualizado: datadir=D:/MySQL/Data" -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 5: Asignar permisos al servicio MySQL" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
icacls $destData /grant "NETWORK SERVICE:(OI)(CI)F" /T | Out-Null
icacls $destData /grant "SYSTEM:(OI)(CI)F" /T | Out-Null
Write-Host "  Permisos asignados." -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 6: Iniciar servicio MySQL80" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Start-Service -Name MySQL80
Start-Sleep -Seconds 5
$svc = Get-Service MySQL80
if ($svc.Status -eq 'Running') {
    Write-Host "  MySQL corriendo. Verificando datadir..." -ForegroundColor Green
    $dir = & $mysql -u root -p123 -e "SELECT @@datadir;" 2>&1 | Where-Object { $_ -notmatch 'Warning' }
    Write-Host "  $dir" -ForegroundColor Yellow
} else {
    Write-Host "  ERROR: MySQL no arranco. Status: $($svc.Status)" -ForegroundColor Red
    Write-Host "  Revisa el Visor de eventos de Windows." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 7: Descargar Sakila (~1 MB)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
$sakilaZip = "$tmpDir\sakila.zip"
Invoke-WebRequest -Uri "https://downloads.mysql.com/docs/sakila-db.zip" -OutFile $sakilaZip -UseBasicParsing
Expand-Archive -Path $sakilaZip -DestinationPath $tmpDir -Force
Write-Host "  Sakila descargada. Instalando schema..." -ForegroundColor Green
Get-Content "$tmpDir\sakila-db\sakila-schema.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
Write-Host "  Instalando datos..." -ForegroundColor Green
Get-Content "$tmpDir\sakila-db\sakila-data.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
Write-Host "  Sakila instalada." -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PASO 8: Descargar World (~90 KB)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
$worldZip = "$tmpDir\world.zip"
Invoke-WebRequest -Uri "https://downloads.mysql.com/docs/world-db.zip" -OutFile $worldZip -UseBasicParsing
Expand-Archive -Path $worldZip -DestinationPath $tmpDir -Force
Write-Host "  World descargada. Instalando..." -ForegroundColor Green
Get-Content "$tmpDir\world-db\world.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
Write-Host "  World instalada." -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  VERIFICACION FINAL" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
& $mysql -u root -p123 -e "SHOW DATABASES;" 2>&1 | Where-Object { $_ -notmatch 'Warning' }

Write-Host ""
Write-Host "  LISTO." -ForegroundColor Green
Write-Host "  - datadir movido a D:\MySQL\Data" -ForegroundColor Green
Write-Host "  - Bases instaladas: sakila, world, demo_db" -ForegroundColor Green
Write-Host "  - Conectate desde DBeaver normalmente (mismo puerto 3306)" -ForegroundColor Green
