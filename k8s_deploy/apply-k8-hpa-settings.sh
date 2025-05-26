#!/bin/bash
set -e

kubectl apply -f llm-k8-configmap.yaml
kubectl apply -f llm-k8-deploy.yaml
kubectl apply -f llm-k8-service.yaml
kubectl apply -f llm-k8-hpa.yaml

echo "Kubernetes resources applied."