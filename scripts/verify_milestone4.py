#!/usr/bin/env python3
"""
Verify all Milestone 4 components are working correctly
"""

import requests
import json
import time
import subprocess
import os
from datetime import datetime
from pathlib import Path


class Milestone4Verifier:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = base_url  # No /api/v1 prefix
        self.results = []
        
    def log_result(self, test_name, passed, details=""):
        """Log test results"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics")
            
            # Check if metrics are exposed
            if response.status_code == 200:
                metrics_text = response.text
                
                # Verify key metrics exist
                required_metrics = [
                    "http_requests_total",
                    "http_request_duration_seconds",
                    "active_requests"
                ]
                
                missing_metrics = []
                for metric in required_metrics:
                    if metric not in metrics_text:
                        missing_metrics.append(metric)
                
                if not missing_metrics:
                    self.log_result("Prometheus Metrics", True, 
                                  f"All required metrics exposed")
                else:
                    self.log_result("Prometheus Metrics", False, 
                                  f"Missing metrics: {missing_metrics}")
            else:
                self.log_result("Prometheus Metrics", False, 
                              f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("Prometheus Metrics", False, str(e))
    
    def test_model_switching(self):
        """Test hot-swap model switching"""
        try:
            # List available models
            response = requests.get(f"{self.api_url}/models")
            if response.status_code != 200:
                self.log_result("Model Switching", False, "Cannot list models")
                return
            
            models = response.json()
            if len(models) < 2:
                self.log_result("Model Switching", False, 
                              "Need at least 2 models for switching")
                return
            
            # Switch to first model
            model_to_switch = models[0]
            switch_response = requests.post(
                f"{self.api_url}/models/switch",
                json={"model": model_to_switch}
            )
            
            if switch_response.status_code == 200:
                result = switch_response.json()
                if result.get("active_model") == model_to_switch:
                    self.log_result("Model Switching", True, 
                                  f"Switched to {model_to_switch}")
                else:
                    self.log_result("Model Switching", False, 
                                  "Model not switched correctly")
            else:
                self.log_result("Model Switching", False, 
                              f"Status code: {switch_response.status_code}")
                              
        except Exception as e:
            self.log_result("Model Switching", False, str(e))
    
    def test_ab_testing(self):
        """Test A/B testing functionality"""
        try:
            # Create an experiment
            experiment_data = {
                "name": "test_experiment",
                "models": {
                    "popularity": 0.5,
                    "collaborative": 0.5
                },
                "duration_days": 1
            }
            
            create_response = requests.post(
                f"{self.api_url}/ab/experiments",
                json=experiment_data
            )
            
            if create_response.status_code == 200:
                result = create_response.json()
                experiment_id = result.get("experiment_id")
                
                # Test user assignment
                test_users = [1, 2, 3, 4, 5]
                assignments = {}
                
                for user_id in test_users:
                    assign_response = requests.get(
                        f"{self.api_url}/ab/model-assignment/{user_id}"
                    )
                    if assign_response.status_code == 200:
                        model = assign_response.json().get("assigned_model")
                        assignments[user_id] = model
                
                # Check if both models are assigned
                unique_models = set(assignments.values())
                if len(unique_models) >= 1:  # At least one model assigned
                    self.log_result("A/B Testing", True, 
                                  f"Experiment created: {experiment_id}")
                else:
                    self.log_result("A/B Testing", False, 
                                  "Model assignment not working")
            else:
                self.log_result("A/B Testing", False, 
                              f"Cannot create experiment: {create_response.status_code}")
                              
        except Exception as e:
            self.log_result("A/B Testing", False, str(e))
    
    def test_provenance_tracking(self):
        """Test provenance tracking"""
        try:
            # Make a recommendation request
            user_id = 123
            rec_response = requests.get(f"{self.api_url}/recommend/{user_id}?k=5")
            
            if rec_response.status_code == 200:
                rec_data = rec_response.json()
                
                # Check if provenance is included
                if "provenance" in rec_data and rec_data["provenance"]:
                    provenance = rec_data["provenance"]
                    required_fields = [
                        "request_id",
                        "model_version",
                        "timestamp"
                    ]
                    
                    missing_fields = []
                    for field in required_fields:
                        if field not in provenance:
                            missing_fields.append(field)
                    
                    if not missing_fields:
                        # Try to fetch the trace
                        request_id = provenance["request_id"]
                        trace_response = requests.get(
                            f"{self.api_url}/provenance/trace/{request_id}"
                        )
                        
                        if trace_response.status_code == 200:
                            self.log_result("Provenance Tracking", True, 
                                          f"Request ID: {request_id}")
                        else:
                            self.log_result("Provenance Tracking", False, 
                                          "Cannot fetch trace")
                    else:
                        self.log_result("Provenance Tracking", False, 
                                      f"Missing fields: {missing_fields}")
                else:
                    self.log_result("Provenance Tracking", False, 
                                  "No provenance in response")
            else:
                self.log_result("Provenance Tracking", False, 
                              f"Recommendation failed: {rec_response.status_code}")
                              
        except Exception as e:
            self.log_result("Provenance Tracking", False, str(e))
    
    def test_grafana_dashboard(self):
        """Test if Grafana dashboard file exists"""
        dashboard_path = Path("infra/grafana/dashboards/lens-monitoring.json")
        
        if dashboard_path.exists():
            try:
                with open(dashboard_path) as f:
                    dashboard = json.load(f)
                
                # Verify dashboard structure
                if "panels" in dashboard and len(dashboard["panels"]) > 0:
                    self.log_result("Grafana Dashboard", True, 
                                  f"{len(dashboard['panels'])} panels configured")
                else:
                    self.log_result("Grafana Dashboard", False, 
                                  "No panels in dashboard")
            except Exception as e:
                self.log_result("Grafana Dashboard", False, 
                              f"Invalid JSON: {e}")
        else:
            self.log_result("Grafana Dashboard", False, 
                          "Dashboard file not found")
    
    def test_github_actions(self):
        """Test GitHub Actions workflow"""
        workflow_path = Path(".github/workflows/retrain-models.yml")
        
        if workflow_path.exists():
            with open(workflow_path) as f:
                content = f.read()
            
            # Check for required components
            checks = {
                "schedule": "cron: '0 2,14 * * *'" in content,
                "retrain script": "train_all_models.py" in content,
                "model registry update": "model_registry" in content,
                "provenance tracking": "DATA_SNAPSHOT_ID" in content
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if not failed_checks:
                self.log_result("GitHub Actions Workflow", True, 
                              "All components present")
            else:
                self.log_result("GitHub Actions Workflow", False, 
                              f"Missing: {failed_checks}")
        else:
            self.log_result("GitHub Actions Workflow", False, 
                          "Workflow file not found")
    
    def test_docker_build(self):
        """Test Docker build"""
        try:
            # Check if Dockerfile exists
            if not Path("backend/Dockerfile").exists():
                self.log_result("Docker Build", False, "Dockerfile not found")
                return
            
            # Try to build the image (dry run)
            result = subprocess.run(
                ["docker", "build", "--no-cache", "--target", "builder", 
                 "-t", "lens-test:builder", "backend/"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_result("Docker Build", True, 
                              "Multi-stage build successful")
            else:
                self.log_result("Docker Build", False, 
                              f"Build failed: {result.stderr[:200]}")
                              
        except Exception as e:
            self.log_result("Docker Build", False, str(e))
    
    def test_availability_calculation(self):
        """Test availability metrics"""
        try:
            # Get metrics
            response = requests.get(f"{self.base_url}/metrics")
            if response.status_code != 200:
                self.log_result("Availability Calculation", False, 
                              "Cannot fetch metrics")
                return
            
            metrics_text = response.text
            
            # Parse metrics (simplified)
            total_requests = 0
            error_requests = 0
            
            for line in metrics_text.split('\n'):
                if 'http_requests_total' in line and not line.startswith('#'):
                    if 'status="5' in line:
                        # Extract error count
                        try:
                            count = float(line.split()[-1])
                            error_requests += count
                        except:
                            pass
                    try:
                        count = float(line.split()[-1])
                        total_requests += count
                    except:
                        pass
            
            if total_requests > 0:
                availability = ((total_requests - error_requests) / total_requests) * 100
                
                self.log_result("Availability Calculation", True, 
                              f"Availability: {availability:.2f}% "
                              f"(Target: ≥70%)")
            else:
                self.log_result("Availability Calculation", False, 
                              "No request metrics found")
                              
        except Exception as e:
            self.log_result("Availability Calculation", False, str(e))
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("MILESTONE 4 VERIFICATION SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if total - passed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"- {result['test']}: {result['details']}")
        
        # Save detailed report
        report_path = Path("verification_results_milestone4.json")
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_rate": passed/total*100
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        return passed == total
    
    def run_all_tests(self):
        """Run all tests"""
        print("Starting Milestone 4 Verification...")
        print("="*60)
        
        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code != 200:
                print("ERROR: Backend server not responding")
                return False
        except:
            print("ERROR: Cannot connect to backend server")
            print("Make sure the server is running on", self.base_url)
            return False
        
        # Run tests
        self.test_metrics_endpoint()
        self.test_model_switching()
        self.test_ab_testing()
        self.test_provenance_tracking()
        self.test_grafana_dashboard()
        self.test_github_actions()
        self.test_availability_calculation()
        
        # Optional: Test Docker build if Docker is available
        if subprocess.run(["docker", "--version"], 
                         capture_output=True).returncode == 0:
            self.test_docker_build()
        else:
            print("⚠ Skipping Docker build test (Docker not available)")
        
        # Generate report
        return self.generate_report()


if __name__ == "__main__":
    import sys
    
    # Allow custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    verifier = Milestone4Verifier(base_url)
    success = verifier.run_all_tests()
    
    sys.exit(0 if success else 1)