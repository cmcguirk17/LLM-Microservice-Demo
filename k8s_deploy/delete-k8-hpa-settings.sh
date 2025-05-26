#!/bin/bash
set -e

kubectl delete -f llm-k8-hpa.yaml
kubectl delete -f llm-k8-service.yaml
kubectl delete -f llm-k8-deploy.yaml
kubectl delete -f llm-k8-configmap.yaml

echo "Kubernetes resources deleted."