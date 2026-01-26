#!/bin/bash
# Generate CA, server, and client certificates for mTLS
# Run this script to set up certificates for the first time

set -e

CERT_DIR="caddy/certs"
mkdir -p "$CERT_DIR"

echo "Generating CA private key..."
openssl genrsa -out "$CERT_DIR/ca.key" 4096

echo "Generating CA certificate..."
openssl req -new -x509 -days 3650 -key "$CERT_DIR/ca.key" -out "$CERT_DIR/ca.crt" \
    -subj "/CN=VoiceTranscriptor CA/O=VoiceTranscriptor/C=US"

echo "Generating server private key..."
openssl genrsa -out "$CERT_DIR/server.key" 4096

echo "Generating server certificate signing request..."
openssl req -new -key "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" \
    -subj "/CN=localhost/O=VoiceTranscriptor Server/C=US"

echo "Generating server certificate (signed by CA)..."
openssl x509 -req -days 365 -in "$CERT_DIR/server.csr" -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" \
    -CAcreateserial -out "$CERT_DIR/server.crt" \
    -extensions v3_req -extfile <(
        echo "[v3_req]"
        echo "keyUsage = keyEncipherment, dataEncipherment"
        echo "extendedKeyUsage = serverAuth"
        echo "subjectAltName = @alt_names"
        echo "[alt_names]"
        echo "DNS.1 = localhost"
        echo "IP.1 = 127.0.0.1"
    )

echo "Generating client private key..."
openssl genrsa -out "$CERT_DIR/client.key" 4096

echo "Generating client certificate signing request..."
openssl req -new -key "$CERT_DIR/client.key" -out "$CERT_DIR/client.csr" \
    -subj "/CN=VoiceTranscriptor Client/O=VoiceTranscriptor Client/C=US"

echo "Generating client certificate (signed by CA)..."
openssl x509 -req -days 365 -in "$CERT_DIR/client.csr" -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" \
    -CAcreateserial -out "$CERT_DIR/client.crt" \
    -extensions v3_req -extfile <(
        echo "[v3_req]"
        echo "keyUsage = digitalSignature, keyEncipherment"
        echo "extendedKeyUsage = clientAuth"
    )

echo "Creating PKCS#12 bundle for Android (.p12 file)..."
openssl pkcs12 -export -out "$CERT_DIR/client.p12" \
    -inkey "$CERT_DIR/client.key" \
    -in "$CERT_DIR/client.crt" \
    -certfile "$CERT_DIR/ca.crt" \
    -passout pass:changeme  # Change this password!

echo ""
echo "=========================================="
echo "Certificates generated successfully!"
echo "=========================================="
echo ""
echo "Files created in $CERT_DIR/:"
echo "  - ca.crt          (CA certificate - install on clients)"
echo "  - server.crt      (Server certificate)"
echo "  - server.key      (Server private key)"
echo "  - client.crt      (Client certificate)"
echo "  - client.key      (Client private key)"
echo "  - client.p12      (PKCS#12 bundle for Android - password: changeme)"
echo ""
echo "IMPORTANT:"
echo "  1. Change the password in client.p12 before using!"
echo "  2. Keep ca.key and server.key SECRET - never share them!"
echo "  3. Distribute ca.crt and client.p12 to authorized users only"
echo ""
