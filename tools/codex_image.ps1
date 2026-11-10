# Generates one image from a ChatGPT/<name>.prompt.txt through the user's
# Codex CLI (built-in image_gen tool, ChatGPT sign-in, no API key) and saves
# it as ChatGPT/<name>.png, the same drop the user makes by hand.
#
#   powershell -File tools/codex_image.ps1 ChatGPT/skill-farming-icon.prompt.txt
#   powershell -File tools/codex_image.ps1 a.prompt.txt b.prompt.txt   # one run each
#
# It always uses the ChatGPT subscription's Codex usage, never API billing:
# the API key variables are cleared and forced_login_method is "chatgpt".
# Codex runs read-only in an empty folder outside the repo, so it sees only
# the prompt (not AGENTS.md) and cannot touch project files; this script
# picks the new PNG out of ~/.codex/generated_images and copies it.
param(
    [Parameter(Mandatory, ValueFromRemainingArguments)] [string[]] $Prompts,
    [switch] $Force,
    [int] $TimeoutSec = 600
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
# The desktop app keeps its current CLI in a hashed bin\<id>\ folder; the
# top-level bin\codex.exe can be a stale build that rejects the current model.
$codex = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'OpenAI\Codex\bin') -Recurse -Filter codex.exe -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $codex) { $codex = (Get-Command codex -ErrorAction Stop).Source }
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
$generated = Join-Path $codexHome 'generated_images'
$work = Join-Path ([IO.Path]::GetTempPath()) 'vlg-codex-image'
New-Item -ItemType Directory -Force $work | Out-Null

foreach ($p in $Prompts) {
    $promptFile = (Resolve-Path (Join-Path $root $p) -ErrorAction SilentlyContinue)
    if (-not $promptFile) { $promptFile = Resolve-Path $p }
    $name = [IO.Path]::GetFileName($promptFile) -replace '\.prompt\.txt$', ''
    $out = Join-Path $root "ChatGPT\$name.png"
    if ((Test-Path $out) -and -not $Force) { Write-Host "skip $name (ChatGPT\$name.png exists; -Force to redo)"; continue }

    $spec = Get-Content $promptFile -Raw -Encoding UTF8
    $instructions = @"
Use your built-in image_gen tool exactly once to generate the image specified below. Do not write code, scripts, SVG or placeholder art, do not use the CLI fallback, and do not edit any files. Follow the specification literally, including transparency and output size. Ignore any 'Project destination' or follow-up steps in it; just generate the image and reply with one line: DONE.

$spec
"@
    $started = Get-Date
    Write-Host "generating $name ..."
    $log = Join-Path $work "$name.log"
    $job = Start-Job -ScriptBlock {
        param($codex, $work, $instructions, $log)
        # Bill the user's ChatGPT subscription only: no API key fallback.
        Remove-Item Env:OPENAI_API_KEY, Env:CODEX_API_KEY -ErrorAction SilentlyContinue
        $instructions | & $codex exec --skip-git-repo-check -s read-only -C $work --color never `
            -c 'forced_login_method="chatgpt"' - *> $log
        $LASTEXITCODE
    } -ArgumentList $codex, $work, $instructions, $log
    if (-not (Wait-Job $job -Timeout $TimeoutSec)) {
        Stop-Job $job; Remove-Job $job -Force
        Write-Warning "$name timed out after $TimeoutSec s (log: $log)"; continue
    }
    $code = Receive-Job $job; Remove-Job $job
    $png = Get-ChildItem $generated -Recurse -Filter *.png -ErrorAction SilentlyContinue |
        Where-Object LastWriteTime -gt $started | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $png) { Write-Warning "$name produced no image (exit $code, log: $log)"; continue }
    Copy-Item $png.FullName $out -Force
    Write-Host "saved ChatGPT\$name.png  (from $($png.FullName))"
}
