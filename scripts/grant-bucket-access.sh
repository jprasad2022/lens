#!/bin/bash
# Grant Cloud Run service account access to the data bucket

PROJECT_ID="940371601491"
BUCKET_NAME="lens-data-940371601491"

# Get the default compute service account
SERVICE_ACCOUNT="${PROJECT_ID}-compute@developer.gserviceaccount.com"

echo "Granting Storage Object Viewer role to: $SERVICE_ACCOUNT"

# Grant read access to the bucket
gsutil iam ch serviceAccount:${SERVICE_ACCOUNT}:objectViewer gs://${BUCKET_NAME}

echo "✅ Access granted! The Cloud Run service can now read from the bucket."