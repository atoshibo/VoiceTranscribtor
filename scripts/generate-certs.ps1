# PowerShell script to generate certificates for mTLS
# Run this on Windows

$CERT_DIR = "caddy\certs"
New-Item -ItemType Directory -Force -Path $CERT_DIR | Out-Null

Write-Host "Generating CA private key..."
openssl genrsa -out "$CERT_DIR\ca.key" 4096

Write-Host "Generating CA certificate..."
openssl req -new -x509 -days 3650 -key "$CERT_DIR\ca.key" -out "$CERT_DIR\ca.crt" `
    -subj "/CN=VoiceTranscriptor CA/O=VoiceTranscriptor/C=US"

Write-Host "Generating server private key..."
openssl genrsa -out "$CERT_DIR\server.key" 4096

Write-Host "Generating server certificate signing request..."
openssl req -new -key "$CERT_DIR\server.key" -out "$CERT_DIR\server.csr" `
    -subj "/CN=localhost/O=VoiceTranscriptor Server/C=US"

Write-Host "Generating server certificate..."
# Create ext file for server cert
$extContent = @"
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
"@
$extContent | Out-File -FilePath "$CERT_DIR\server.ext" -Encoding ASCII

openssl x509 -req -days 365 -in "$CERT_DIR\server.csr" -CA "$CERT_DIR\ca.crt" -CAkey "$CERT_DIR\ca.key" `
    -CAcreateserial -out "$CERT_DIR\server.crt" -extensions v3_req -extfile "$CERT_DIR\server.ext"

Write-Host "Generating client private key..."
openssl genrsa -out "$CERT_DIR\client.key" 4096

Write-Host "Generating client certificate signing request..."
openssl req -new -key "$CERT_DIR\client.key" -out "$CERT_DIR\client.csr" `
    -subj "/CN=VoiceTranscriptor Client/O=VoiceTranscriptor Client/C=US"

Write-Host "Generating client certificate..."
# Create ext file for client cert
$clientExtContent = @"
[v3_req]
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
"@
$clientExtContent | Out-File -FilePath "$CERT_DIR\client.ext" -Encoding ASCII

openssl x509 -req -days 365 -in "$CERT_DIR\client.csr" -CA "$CERT_DIR\ca.crt" -CAkey "$CERT_DIR\ca.key" `
    -CAcreateserial -out "$CERT_DIR\client.crt" -extensions v3_req -extfile "$CERT_DIR\client.ext"

Write-Host "Creating PKCS#12 bundle for Android..."
openssl pkcs12 -export -out "$CERT_DIR\client.p12" `
    -inkey "$CERT_DIR\client.key" `
    -in "$CERT_DIR\client.crt" `
    -certfile "$CERT_DIR\ca.crt" `
    -passout pass:changeme

Write-Host ""
Write-Host "=========================================="
Write-Host "Certificates generated successfully!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Files created in $CERT_DIR\:"
Write-Host "  - ca.crt          (CA certificate)"
Write-Host "  - server.crt      (Server certificate)"
Write-Host "  - server.key      (Server private key)"
Write-Host "  - client.crt      (Client certificate)"
Write-Host "  - client.key      (Client private key)"
Write-Host "  - client.p12      (PKCS#12 bundle for Android - password: changeme)"
Write-Host ""
Write-Host "IMPORTANT:"
Write-Host "  1. Change the password in client.p12 before using!"
Write-Host "  2. Keep ca.key and server.key SECRET!"
Write-Host "  3. Distribute ca.crt and client.p12 to authorized users only"
Write-Host ""
