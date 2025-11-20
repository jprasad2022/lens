#!/usr/bin/env python3
"""
Collect all Milestone 4 evidence - Windows compatible version
"""
import subprocess
import json
import os
import sys
import platform
from datetime import datetime
import time
import requests

def run_command(cmd, description, windows_cmd=None):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    
    # Use Windows-specific command if provided and on Windows
    if platform.system() == "Windows" and windows_cmd:
        cmd = windows_cmd
    
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        return result.stdout
    except Exception as e:
        print(f"Failed: {e}")
        return None

def check_docker_windows():
    """Check Docker on Windows"""
    print("\n============================================================")
    print("Running: Docker containers")
    print("Command: docker ps")
    print("============================================================")
    
    try:
        # Use simple docker ps command
        result = subprocess.run("docker ps", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return result.stdout
        else:
            print(f"Docker command failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Docker check failed: {e}")
        return None

def check_model_files_windows():
    """Check model files on Windows"""
    print("\n============================================================")
    print("Running: Model registry contents")
    print("Command: Listing model files")
    print("============================================================")
    
    model_dir = "backend/model_registry"
    if os.path.exists(model_dir):
        output = []
        for root, dirs, files in os.walk(model_dir):
            # Print directory structure
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                path = os.path.join(root, file)
                size = os.path.getsize(path)
                print(f"{subindent}{file} ({size:,} bytes)")
                output.append(f"{path} ({size} bytes)")
        return "\n".join(output) if output else "No model files found"
    return "Model registry directory not found"

def check_metrics_windows():
    """Check metrics endpoint on Windows"""
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            # Return first 20 lines
            lines = response.text.split('\n')[:20]
            return '\n'.join(lines)
    except Exception as e:
        print(f"Metrics check failed: {e}")
    return None

def run_python_script(script_path, description):
    """Run a Python script using current Python interpreter"""
    if os.path.exists(script_path):
        python_cmd = sys.executable  # Use current Python interpreter
        
        # Add delay for rate-limited scripts
        if "provenance" in script_path:
            print("Waiting 60 seconds before provenance test...")
            time.sleep(60)
        elif "availability" in script_path:
            print("Waiting 30 seconds before availability test...")
            time.sleep(30)
        
        cmd = f'"{python_cmd}" "{script_path}"'
        return run_command(cmd, description)
    else:
        print(f"Script not found: {script_path}")
        return None

def main():
    """Collect all evidence"""
    evidence = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "tasks": {}
    }
    
    print("MILESTONE 4 EVIDENCE COLLECTION (Windows)")
    print("="*60)
    
    # 1. Check Docker deployment
    print("\n1. CONTAINERIZATION EVIDENCE")
    docker_output = check_docker_windows()
    evidence["tasks"]["containerization"] = {
        "docker_ps": docker_output,
        "backend_running": "lens-backend" in docker_output if docker_output else False
    }
    
    # 2. Model retraining (local simulation)
    print("\n2. MODEL RETRAINING EVIDENCE")
    model_files = check_model_files_windows()
    evidence["tasks"]["retraining"] = {
        "model_files": model_files,
        "workflow_file": os.path.exists(".github/workflows/retrain-models.yml")
    }
    
    # 3. Monitoring check
    print("\n3. MONITORING EVIDENCE")
    metrics = check_metrics_windows()
    evidence["tasks"]["monitoring"] = {
        "metrics_available": metrics is not None and "http_requests_total" in metrics if metrics else False,
        "grafana_url": "http://localhost:3001",
        "metrics_sample": metrics[:500] if metrics else None
    }
    
    # 4. A/B Testing
    print("\n4. A/B TESTING EVIDENCE")
    print("Running A/B experiment...")
    ab_result = run_python_script("scripts/run_ab_experiment.py", 
                                  "A/B experiment with statistical analysis")
    
    # Add longer delay after A/B test to avoid rate limiting
    print("\nWaiting 60 seconds for rate limit to reset...")
    time.sleep(60)
    evidence["tasks"]["ab_testing"] = {
        "experiment_run": ab_result is not None,
        "statistical_test": "t-test" in ab_result if ab_result else False
    }
    
    # 5. Provenance
    print("\n5. PROVENANCE EVIDENCE")
    prov_result = run_python_script("scripts/test_provenance.py", 
                                   "Provenance tracking test")
    evidence["tasks"]["provenance"] = {
        "tracking_enabled": prov_result is not None
    }
    
    # 6. Availability
    print("\n6. AVAILABILITY EVIDENCE")
    avail_result = run_python_script("scripts/calculate_availability.py", 
                                    "Availability calculation")
    evidence["tasks"]["availability"] = {
        "calculation_done": avail_result is not None,
        "meets_slo": "PASSED" in avail_result if avail_result else False
    }
    
    # Save evidence
    with open("milestone4_evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)
    
    print("\n" + "="*60)
    print("EVIDENCE COLLECTION COMPLETE")
    print("="*60)
    print("\nSummary:")
    for task, result in evidence["tasks"].items():
        status = "✓" if any(v for v in result.values() if v) else "✗"
        print(f"{status} {task}")
    
    print("\nEvidence saved to: milestone4_evidence.json")
    print("\nNext steps:")
    print("1. Take screenshots of:")
    print("   - Grafana dashboard at http://localhost:3001")
    print("   - Terminal output from this script")
    print("2. Generate final report: python scripts/generate_milestone4_report.py")
    print("3. Convert report to PDF")

if __name__ == "__main__":
    main()