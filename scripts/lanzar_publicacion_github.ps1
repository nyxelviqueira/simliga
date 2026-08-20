param(
    [string]$Temporada = "2026-27",
    [int]$Sims = 20000,
    [switch]$Esperar
)

$ErrorActionPreference = "Stop"

$owner = "nyxelviqueira"
$repo = "simliga"
$workflow = "publish-panel.yml"
$branch = "main"
$pagesUrl = "https://nyxelviqueira.github.io/simliga/"
$token = $env:SIMLIGA_GITHUB_TOKEN

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host ""
    Write-Host "Falta la variable de entorno SIMLIGA_GITHUB_TOKEN."
    Write-Host ""
    Write-Host "Creala una vez con un token Fine-grained de GitHub:"
    Write-Host "  - Repository access: nyxelviqueira/simliga"
    Write-Host "  - Repository permissions: Actions = Read and write"
    Write-Host ""
    Write-Host "Despues guardalo en Windows con:"
    Write-Host '  setx SIMLIGA_GITHUB_TOKEN "github_pat_TU_TOKEN"'
    Write-Host ""
    Write-Host "Cierra esta terminal y abre una nueva antes de volver a ejecutar."
    exit 1
}

$headers = @{
    Accept = "application/vnd.github+json"
    Authorization = "Bearer $token"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent" = "SimLiga"
}

$dispatchUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$workflow/dispatches"
$startedAt = [DateTime]::UtcNow.AddSeconds(-10)
$body = @{
    ref = $branch
    inputs = @{
        temporada = $Temporada
        sims = [string]$Sims
    }
} | ConvertTo-Json -Depth 5

Write-Host "Lanzando workflow de GitHub Actions..."

try {
    Invoke-RestMethod -Method Post -Uri $dispatchUrl -Headers $headers -ContentType "application/json" -Body $body | Out-Null
} catch {
    $status = $null
    if ($_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
    }
    Write-Host ""
    Write-Host "GitHub ha rechazado la peticion."
    if ($status) {
        Write-Host "Codigo HTTP: $status"
    }
    Write-Host "Comprueba que el token tenga permiso Actions = Read and write para $owner/$repo."
    throw
}

Write-Host "Workflow lanzado."

if (-not $Esperar) {
    Write-Host "Mira el progreso en: https://github.com/$owner/$repo/actions"
    exit 0
}

$runsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$workflow/runs?event=workflow_dispatch&branch=$branch&per_page=10"
$run = $null

Write-Host "Buscando la ejecucion nueva..."
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    $runs = Invoke-RestMethod -Method Get -Uri $runsUrl -Headers $headers
    $run = $runs.workflow_runs |
        Where-Object { ([DateTime]$_.created_at).ToUniversalTime() -ge $startedAt } |
        Sort-Object created_at -Descending |
        Select-Object -First 1
    if ($run) {
        break
    }
}

if (-not $run) {
    Write-Host "No he encontrado el run nuevo todavia."
    Write-Host "Revisalo aqui: https://github.com/$owner/$repo/actions"
    exit 0
}

Write-Host "Run: $($run.html_url)"
Write-Host "Esperando a que termine..."

while ($run.status -ne "completed") {
    Start-Sleep -Seconds 15
    $run = Invoke-RestMethod -Method Get -Uri $run.url -Headers $headers
    Write-Host "Estado: $($run.status)"
}

if ($run.conclusion -eq "success") {
    Write-Host ""
    Write-Host "Publicado correctamente."
    Write-Host "Panel movil: $pagesUrl"
    exit 0
}

Write-Host ""
Write-Host "El workflow ha terminado con estado: $($run.conclusion)"
Write-Host "Abre el log aqui: $($run.html_url)"
exit 1
