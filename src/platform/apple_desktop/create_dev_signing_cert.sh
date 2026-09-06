#!/usr/bin/env bash
# Create a self-signed code signing certificate for Bad Apple so Accessibility
# (TCC) grants survive rebuilds. The certificate is imported into a dedicated
# keychain with an empty password, so it does not require the user's login
# keychain password.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CERT_NAME="Bad Apple Dev (local)"
CERT_FILE="${REPO_ROOT}/.badapple_dev_cert.p12"
KEY_FILE="${REPO_ROOT}/.badapple_dev_cert.key"
CRT_FILE="${REPO_ROOT}/.badapple_dev_cert.crt"
KEYCHAIN="${HOME}/Library/Keychains/badapple.keychain-db"
CN="Bad Apple Dev (local)"

if [[ -f "${CERT_FILE}" ]]; then
    echo "Developer certificate already exists: ${CERT_FILE}"
    exit 0
fi

CONFIG="$(mktemp)"
cat > "${CONFIG}" <<'CONF'
[ req ]
distinguished_name = req_distinguished_name
prompt = no
x509_extensions = v3_req

[ req_distinguished_name ]
O = Bad Apple Dev
CN = Bad Apple Dev (local)

[ v3_req ]
keyUsage = critical, digitalSignature
extendedKeyUsage = codeSigning
CONF

openssl req -x509 -newkey rsa:2048 -keyout "${KEY_FILE}" -out "${CRT_FILE}" -nodes -days 3650 -config "${CONFIG}"
rm "${CONFIG}"

P12_PASS="badapple"
openssl pkcs12 -export -in "${CRT_FILE}" -inkey "${KEY_FILE}" -out "${CERT_FILE}" -name "${CERT_NAME}" -passout pass:"${P12_PASS}"

# Create a dedicated, unlocked keychain with an empty password.
if [[ ! -f "${KEYCHAIN}" ]]; then
    security create-keychain -p "" "${KEYCHAIN}"
fi
security unlock-keychain -p "" "${KEYCHAIN}" || true

# Import the cert and allow codesign to use it without prompting.
security import "${CERT_FILE}" -k "${KEYCHAIN}" -P "${P12_PASS}" -T /usr/bin/codesign -T /usr/bin/security
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" "${KEYCHAIN}" || true

# Add the dedicated keychain to the user's search list.
CURRENT=$(security list-keychains | tr -d '"' | tr '\n' ' ')
security list-keychains -s ${CURRENT} "${KEYCHAIN}"

echo "Created developer certificate: ${CERT_NAME}"
echo "Certificate file: ${CERT_FILE}"
echo "Keychain: ${KEYCHAIN}"
echo ""
echo "To make the cert visible in System Settings as trusted (optional), run:"
echo "  security add-trusted-cert -r trustRoot -k ${KEYCHAIN} ${CRT_FILE}"
