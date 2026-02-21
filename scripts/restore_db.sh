#!/usr/bin/env bash
# Restore database from GCS backup
# Usage: ./scripts/restore_db.sh [backup_filename]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
CONTAINER="deepsummit_postgres"
BACKUP_DIR="$PROJECT_ROOT/backups"
DATA_DIR="$PROJECT_ROOT/data/postgres"

# Load environment
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found"
    exit 1
fi
set -a && source "$ENV_FILE" && set +a

# Check required vars
for var in POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB GCP_BUCKET_NAME; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var not set in .env"
        exit 1
    fi
done

# Check gcloud auth
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "ERROR: Not authenticated. Run: gcloud auth login"
    exit 1
fi

# List available backups
echo "Available backups:"
BACKUPS=$(gsutil ls "gs://$GCP_BUCKET_NAME/local-dev/*.sql.gz" 2>/dev/null || echo "")
if [ -z "$BACKUPS" ]; then
    echo "No backups found. Create one with: ./scripts/backup_db.sh"
    exit 1
fi
gsutil ls -lh "gs://$GCP_BUCKET_NAME/local-dev/*.sql.gz"

# Select backup
if [ $# -eq 1 ]; then
    BACKUP_FILE="gs://$GCP_BUCKET_NAME/local-dev/$1"
else
    echo ""
    read -p "Enter backup filename (or press Enter for latest): " BACKUP_NAME
    if [ -z "$BACKUP_NAME" ]; then
        BACKUP_FILE=$(gsutil ls "gs://$GCP_BUCKET_NAME/local-dev/*.sql.gz" | sort | tail -n 1)
    else
        BACKUP_FILE="gs://$GCP_BUCKET_NAME/local-dev/$BACKUP_NAME"
    fi
fi

echo ""
echo "Selected: $BACKUP_FILE"
echo ""
echo "WARNING: This will delete all existing data!"
read -p "Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled"
    exit 0
fi

# Download backup
mkdir -p "$BACKUP_DIR"
FILENAME=$(basename "$BACKUP_FILE")
LOCAL_PATH="$BACKUP_DIR/$FILENAME"
echo "Downloading backup..."
gsutil cp "$BACKUP_FILE" "$LOCAL_PATH"

# Stop container if running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Stopping container..."
    docker stop "$CONTAINER" > /dev/null
fi

# Remove old data
if [ -d "$DATA_DIR" ]; then
    echo "Removing old database data..."
    rm -rf "$DATA_DIR"
fi

# Start fresh container
echo "Starting fresh container..."
docker compose up -d postgres > /dev/null

# Wait for postgres to be ready
echo "Waiting for PostgreSQL..."
until docker exec "$CONTAINER" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" &> /dev/null; do
    sleep 1
done

# Restore data
echo "Restoring data..."
gunzip -c "$LOCAL_PATH" | docker exec -i "$CONTAINER" psql \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    > /dev/null 2>&1

# Verify
echo ""
echo "Restored! Row counts:"
docker exec -t "$CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
    SELECT 'Peaks: ' || COUNT(*) FROM peaks
    UNION ALL SELECT 'Expeditions: ' || COUNT(*) FROM expeditions
    UNION ALL SELECT 'Members: ' || COUNT(*) FROM expedition_members;
" 2>/dev/null || true

echo ""
echo "Done! Database restored from: $FILENAME"
