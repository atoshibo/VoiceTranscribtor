# Generate self-signed certificate for HTTPS
# This script creates a simple certificate for local/LAN use

$certDir = "caddy\certs"
if (-not (Test-Path $certDir)) {
    New-Item -ItemType Directory -Path $certDir -Force | Out-Null
}

# Get LAN IP
$lanIP = (Get-NetIPAddress -AddressFamily IPv4 | 
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | 
    Select-Object -First 1).IPAddress

Write-Host "Generating self-signed certificate..."
Write-Host "LAN IP: $lanIP"

# Check if OpenSSL is available
$opensslPath = Get-Command openssl -ErrorAction SilentlyContinue
if ($opensslPath) {
    Write-Host "Using OpenSSL..."
    $san = "subjectAltName=IP:127.0.0.1,IP:$lanIP,DNS:localhost"
    openssl req -x509 -newkey rsa:4096 -keyout "$certDir\server.key" `
        -out "$certDir\server.crt" -days 365 -nodes `
        -subj "/CN=localhost" -addext $san
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Certificate generated successfully!"
        Write-Host "Files created:"
        Write-Host "  - $certDir\server.crt"
        Write-Host "  - $certDir\server.key"
    } else {
        Write-Host "Error generating certificate with OpenSSL"
        exit 1
    }
} else {
    Write-Host "OpenSSL not found. Using PowerShell certificate generation..."
    
    # Use PowerShell to create certificate
    $cert = New-SelfSignedCertificate `
        -DnsName "localhost", $lanIP `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -KeyAlgorithm RSA `
        -KeyLength 4096 `
        -NotAfter (Get-Date).AddYears(1) `
        -FriendlyName "VoiceTranscriptor Server"
    
    # Export certificate
    $certPath = "Cert:\CurrentUser\My\$($cert.Thumbprint)"
    $pwd = ConvertTo-SecureString -String "temp" -Force -AsPlainText
    
    # Export to PFX first
    $pfxPath = "$certDir\temp.pfx"
    Export-PfxCertificate -Cert $certPath -FilePath $pfxPath -Password $pwd | Out-Null
    
    # Extract certificate and key using certutil (Windows built-in)
    Write-Host "Extracting certificate and key..."
    
    # Export certificate
    certutil -exportPFX -p "temp" "$pfxPath" "$certDir\temp.pfx" | Out-Null
    
    # For Windows, we'll use a different approach - create PEM files manually
    # This is a workaround since Windows doesn't have easy OpenSSL-like tools
    Write-Host ""
    Write-Host "WARNING: PowerShell certificate generation is complex."
    Write-Host "Please install OpenSSL for Windows or use WSL:"
    Write-Host ""
    Write-Host "Option 1: Install OpenSSL for Windows"
    Write-Host "  Download from: https://slproweb.com/products/Win32OpenSSL.html"
    Write-Host ""
    Write-Host "Option 2: Use WSL (if available)"
    Write-Host "  wsl openssl req -x509 -newkey rsa:4096 -keyout caddy/certs/server.key \"
    Write-Host "      -out caddy/certs/server.crt -days 365 -nodes \"
    Write-Host "      -subj '/CN=localhost' \"
    Write-Host "      -addext 'subjectAltName=IP:127.0.0.1,IP:$lanIP,DNS:localhost'"
    Write-Host ""
    Write-Host "Option 3: Use Docker to generate certificate"
    Write-Host "  docker run --rm -v ${PWD}/caddy/certs:/certs alpine/openssl req -x509 \"
    Write-Host "      -newkey rsa:4096 -keyout /certs/server.key \"
    Write-Host "      -out /certs/server.crt -days 365 -nodes \"
    Write-Host "      -subj '/CN=localhost' \"
    Write-Host "      -addext 'subjectAltName=IP:127.0.0.1,IP:$lanIP,DNS:localhost'"
    
    # Clean up
    Remove-Item $pfxPath -ErrorAction SilentlyContinue
    Remove-Item $certPath -ErrorAction SilentlyContinue
    
    exit 1
}
