# Alarm Normaliser — TMF642 Multi-Vendor Pipeline

A production-grade alarm normalisation pipeline based on TM Forum TMF642 v4.0 and ITU-T X.733.
Ingests raw alarms from Cisco, Nokia, Ericsson, Huawei, Prometheus, and Kubernetes
and emits a single canonical TMF642-compliant alarm stream.

## Quick start

    git clone https://github.com/YOUR_USERNAME/alarm-normalizer.git
    cd alarm-normalizer
    python tests/test_pipeline.py
    python demo/run_demo.py --scenario fiber

## Requirements

Python 3.8 or higher. No external dependencies.