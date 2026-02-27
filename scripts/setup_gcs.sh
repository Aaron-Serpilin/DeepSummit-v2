#!/usr/bin/env bash
# One-time GCS bucket setup for database backups

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

echo "DeepSummit - GCS Backup Setup"
echo ""

# Check gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "ERROR: Not authenticated. Run: gcloud auth login"
    exit 1
fi

echo "Available GCP projects:"
gcloud projects list --format="table(projectId,name)"
echo ""

read -p "Enter GCP project ID: " PROJECT_ID
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: Project ID required"
    exit 1
fi

if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
    echo "ERROR: Project not found or no access"
    exit 1
fi

gcloud config set project "$PROJECT_ID" --quiet

# Bucket configuration
DEFAULT_BUCKET="deepsummit-db-backups"
read -p "Bucket name [$DEFAULT_BUCKET]: " BUCKET_NAME
BUCKET_NAME="${BUCKET_NAME:-$DEFAULT_BUCKET}"

read -p "Bucket location [europe-west4]: " LOCATION
LOCATION="${LOCATION:-europe-west4}"

read -p "Backup retention days [90]: " RETENTION_DAYS
RETENTION_DAYS="${RETENTION_DAYS:-90}"

echo ""
echo "Creating bucket: $BUCKET_NAME in $LOCATION"

# Create bucket if it doesn't exist
if gsutil ls -b "gs://$BUCKET_NAME" &> /dev/null; then
    echo "Bucket already exists, using it"
else
    gsutil mb -p "$PROJECT_ID" -l "$LOCATION" -b on "gs://$BUCKET_NAME"
fi

# Set lifecycle policy (auto-delete old backups)
LIFECYCLE_CONFIG=$(mktemp)
cat > "$LIFECYCLE_CONFIG" <<EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": $RETENTION_DAYS, "matchesPrefix": ["local-dev/"]}
    }]
  }
}
EOF
gsutil lifecycle set "$LIFECYCLE_CONFIG" "gs://$BUCKET_NAME" 2>/dev/null || true
rm "$LIFECYCLE_CONFIG"

# Update .env file
if [ -f "$ENV_FILE" ]; then
    if grep -q "^GCP_BUCKET_NAME=" "$ENV_FILE"; then
        sed -i.bak "s|^GCP_BUCKET_NAME=.*|GCP_BUCKET_NAME=$BUCKET_NAME|" "$ENV_FILE"
        sed -i.bak "s|^GCP_PROJECT_ID=.*|GCP_PROJECT_ID=$PROJECT_ID|" "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
    else
        echo "" >> "$ENV_FILE"
        echo "GCP_BUCKET_NAME=$BUCKET_NAME" >> "$ENV_FILE"
        echo "GCP_PROJECT_ID=$PROJECT_ID" >> "$ENV_FILE"
    fi
fi

echo ""
echo "Done! Bucket: gs://$BUCKET_NAME"
echo ""
echo "Next steps:"
echo "  1. Start Docker: docker compose up -d"
echo "  2. Test backup: ./scripts/backup_db.sh"
