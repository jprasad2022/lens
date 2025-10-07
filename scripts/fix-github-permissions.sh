#!/bin/bash

# Script to fix GitHub Actions service account permissions for Cloud Run

PROJECT_ID="your-project-id"  # Replace with your actual project ID
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Granting Cloud Run permissions to: $SERVICE_ACCOUNT"

# Grant Cloud Run Admin role (includes all necessary permissions)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.admin"

# Grant Service Account User role (to act as compute service account)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/iam.serviceAccountUser"

# Grant Artifact Registry Reader (to pull Docker images)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/artifactregistry.reader"

echo "Permissions granted successfully!"
echo ""
echo "The service account now has:"
echo "- Cloud Run Admin (deploy, update, delete services)"
echo "- Service Account User (act as compute service account)"
echo "- Artifact Registry Reader (pull Docker images)"