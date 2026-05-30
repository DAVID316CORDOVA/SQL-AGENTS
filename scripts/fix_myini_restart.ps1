# fix_myini_restart.ps1
# Reescribe my.ini sin BOM y reinicia MySQL
# Ejecutar como Administrador

$myini = 'C:\ProgramData\MySQL\MySQL Server 8.0\my.ini'

Write-Host "=== Leyendo my.ini ===" -ForegroundColor Cyan
$txt = [System.IO.File]::ReadAllText($myini)

Write-Host "=== Reescribiendo sin BOM ===" -ForegroundColor Cyan
# UTF8 sin BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($myini, $txt, $utf8NoBom)

# Verificar
$bytes = [System.IO.File]::ReadAllBytes($myini)
Write-Host "  Primeros 3 bytes: $($bytes[0]) $($bytes[1]) $($bytes[2])"
if ($bytes[0] -eq 239) {
    Write-Host "  ERROR: BOM sigue presente." -ForegroundColor Red
    exit 1
} else {
    Write-Host "  OK: Sin BOM." -ForegroundColor Green
}

# Verificar que datadir este bien
$linea = $txt.Split("`n") | Where-Object { $_ -match 'datadir' }
Write-Host "  datadir detectado: $linea" -ForegroundColor Yellow

Write-Host ""
Write-Host "=== Iniciando MySQL80 ===" -ForegroundColor Cyan
Start-Service -Name MySQL80
Start-Sleep -Seconds 6

$svc = Get-Service MySQL80
Write-Host "  Estado: $($svc.Status)" -ForegroundColor $(if ($svc.Status -eq 'Running') { 'Green' } else { 'Red' })

if ($svc.Status -eq 'Running') {
    Write-Host ""
    Write-Host "  MySQL corriendo. Verificando datadir y bases..." -ForegroundColor Green
    $mysql = 'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe'
    & $mysql -u root -p123 -e "SELECT @@datadir; SHOW DATABASES;" 2>&1 | Where-Object { $_ -notmatch 'Warning' }

    Write-Host ""
    Write-Host "=== Instalando Sakila ===" -ForegroundColor Cyan
    $tmpDir = 'C:\Users\DAVID\Desktop\agent_skills\scripts\mysql_downloads'
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

    $sakilaZip = "$tmpDir\sakila.zip"
    if (!(Test-Path "$tmpDir\sakila-db\sakila-schema.sql")) {
        Write-Host "  Descargando Sakila..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://downloads.mysql.com/docs/sakila-db.zip" -OutFile $sakilaZip -UseBasicParsing
        Expand-Archive -Path $sakilaZip -DestinationPath $tmpDir -Force
    }
    Write-Host "  Instalando schema..." -ForegroundColor Yellow
    Get-Content "$tmpDir\sakila-db\sakila-schema.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
    Write-Host "  Instalando datos..." -ForegroundColor Yellow
    Get-Content "$tmpDir\sakila-db\sakila-data.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
    Write-Host "  Sakila lista." -ForegroundColor Green

    Write-Host ""
    Write-Host "=== Instalando World ===" -ForegroundColor Cyan
    $worldZip = "$tmpDir\world.zip"
    if (!(Test-Path "$tmpDir\world-db\world.sql")) {
        Write-Host "  Descargando World..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://downloads.mysql.com/docs/world-db.zip" -OutFile $worldZip -UseBasicParsing
        Expand-Archive -Path $worldZip -DestinationPath $tmpDir -Force
    }
    Write-Host "  Instalando..." -ForegroundColor Yellow
    Get-Content "$tmpDir\world-db\world.sql" | & $mysql -u root -p123 2>&1 | Where-Object { $_ -notmatch 'Warning' }
    Write-Host "  World lista." -ForegroundColor Green

    Write-Host ""
    Write-Host "=== VERIFICACION FINAL ===" -ForegroundColor Cyan
    & $mysql -u root -p123 -e "SHOW DATABASES;" 2>&1 | Where-Object { $_ -notmatch 'Warning' }

    Write-Host ""
    Write-Host "LISTO. datadir en D:\MySQL\Data" -ForegroundColor Green
    Write-Host "Bases instaladas: demo_db, sakila, world" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "MySQL sigue sin arrancar. Mostrando log..." -ForegroundColor Red
    $log = "D:\MySQL\Data\LAPTOP-77S4KBND.err"
    if (Test-Path $log) {
        Get-Content $log -Tail 20
    }
}
