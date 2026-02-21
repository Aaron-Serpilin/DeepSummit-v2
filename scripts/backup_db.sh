#!/usr/bin/env bash
# Backup database to GCS
# Usage: ./scripts/backup_db.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
CONTAINER="deepsummit_postgres"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Load environment
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env first"
    exit 1
fi
set -a && source "$ENV_FILE" && set +a

# Check required vars
for var in POSTGRES_USER POSTGRES_DB GCP_BUCKET_NAME; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var not set in .env"
        exit 1
    fi
done

# Check Docker container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container not running. Run: docker compose up -d"
    exit 1
fi

# Check gcloud auth
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "ERROR: Not authenticated. Run: gcloud auth login"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

# Create backup
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FILENAME="deepsummit_backup_${TIMESTAMP}.sql.gz"
LOCAL_PATH="$BACKUP_DIR/$FILENAME"
GCS_PATH="gs://$GCP_BUCKET_NAME/local-dev/$FILENAME"

echo "Creating backup: $FILENAME"

docker exec -t "$CONTAINER" pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --format=plain \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    | gzip > "$LOCAL_PATH"

BACKUP_SIZE=$(du -h "$LOCAL_PATH" | cut -f1)
echo "Local backup: $LOCAL_PATH ($BACKUP_SIZE)"

# Upload to GCS
echo "Uploading to GCS..."
gsutil cp "$LOCAL_PATH" "$GCS_PATH"
echo "Uploaded: $GCS_PATH"

# Show database stats
echo ""
echo "Database stats:"
docker exec -t "$CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
    SELECT 'Peaks: ' || COUNT(*) FROM peaks
    UNION ALL SELECT 'Expeditions: ' || COUNT(*) FROM expeditions
    UNION ALL SELECT 'Members: ' || COUNT(*) FROM expedition_members;
" 2>/dev/null || true

# Clean old local backups (keep last 3)
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/*.sql.gz 2>/dev/null | wc -l | tr -d ' ')
if [ "$BACKUP_COUNT" -gt 3 ]; then
    ls -1t "$BACKUP_DIR"/*.sql.gz | tail -n +4 | xargs rm -f
    echo "Cleaned old local backups (kept 3)"
fi

echo ""
echo "Done! You can now run: docker compose down"
