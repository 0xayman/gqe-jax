#!/usr/bin/env bash
# Edit this file to change the VM instance configuration used by the helper scripts.
#gcloud compute ssh --zone "us-central1-f" "instance-20260503-104529" --project "aymantarig-project"
# Representation: gcloud compute ssh --zone "us-central1-f" "instance-20260503-143558" --project "aymantarig-project"
# instance-4-qubits
: "${PROJECT_ID:=aymantarig-project}"
: "${ZONE:=us-central1-f}"
: "${INSTANCE:=instance-20260503-104529}"
: "${VM_USER:=aymantarig_aims_ac_za}"
: "${REMOTE_DIR:=~/work/gqe-torch}"
: "${LOCAL_PROJECT_DIR:=$HOME/Downloads/Quantum/project/gqe-torch}"
: "${VM_HOST:=${VM_USER}@${INSTANCE}.${ZONE}.${PROJECT_ID}}"
: "${VM_PATH:=$REMOTE_DIR}"
