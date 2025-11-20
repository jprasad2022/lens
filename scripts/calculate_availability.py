#!/usr/bin/env python3
"""
Calculate API availability from Prometheus metrics
"""
import requests
from datetime import datetime, timedelta
import json

PROMETHEUS_URL = "http://localhost:9090"
BACKEND_URL = "http://localhost:8000"

def query_prometheus(query, start_time=None, end_time=None):
    """Query Prometheus for metrics"""
    if start_time and end_time:
        # Range query
        params = {
            'query': query,
            'start': start_time.isoformat() + 'Z',
            'end': end_time.isoformat() + 'Z',
            'step': '1h'
        }
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
    else:
        # Instant query
        params = {'query': query}
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def calculate_availability_from_metrics():
    """Calculate availability from current metrics"""
    print("="*60)
    print("AVAILABILITY CALCULATION")
    print("="*60)
    
    # Get current metrics from backend
    response = requests.get(f"{BACKEND_URL}/metrics")
    if response.status_code != 200:
        print("ERROR: Cannot fetch metrics from backend")
        return
    
    metrics_text = response.text
    
    # Parse metrics
    total_requests = 0
    error_requests = 0
    
    for line in metrics_text.split('\n'):
        if 'http_requests_total' in line and not line.startswith('#'):
            try:
                # Extract value
                parts = line.split()
                if len(parts) > 0:
                    value = float(parts[-1])
                    
                    # Check if it's an error
                    if 'status="5' in line:
                        error_requests += value
                    
                    total_requests += value
            except:
                pass
    
    print(f"\nCurrent Metrics:")
    print(f"  Total Requests: {int(total_requests)}")
    print(f"  5xx Errors: {int(error_requests)}")
    
    if total_requests > 0:
        availability = ((total_requests - error_requests) / total_requests) * 100
        print(f"  Availability: {availability:.2f}%")
        print(f"  SLO Target: ≥70%")
        print(f"  Status: {'✓ PASSED' if availability >= 70 else '✗ FAILED'}")
    else:
        print("  No requests recorded yet")
        availability = 100.0
    
    # Calculate for different time windows
    print("\n" + "-"*40)
    print("Availability Windows:")
    
    # Submission time (example)
    submission_time = datetime.utcnow()
    
    # 72h before submission
    start_72h_before = submission_time - timedelta(hours=72)
    print(f"\n72h before submission:")
    print(f"  Start: {start_72h_before.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  End: {submission_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Duration: 72 hours")
    
    # 144h after submission
    end_144h_after = submission_time + timedelta(hours=144)
    print(f"\n144h after submission:")
    print(f"  Start: {submission_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  End: {end_144h_after.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Duration: 144 hours")
    
    # Full window
    print(f"\nFull evaluation window:")
    print(f"  Start: {start_72h_before.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  End: {end_144h_after.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Total Duration: 216 hours (9 days)")
    
    # Generate availability report
    report = {
        "calculation_timestamp": datetime.utcnow().isoformat(),
        "current_metrics": {
            "total_requests": int(total_requests),
            "error_requests": int(error_requests),
            "availability_percentage": round(availability, 2)
        },
        "evaluation_window": {
            "submission_time": submission_time.isoformat(),
            "start_time": start_72h_before.isoformat(),
            "end_time": end_144h_after.isoformat(),
            "duration_hours": 216
        },
        "slo": {
            "target": 70.0,
            "achieved": availability >= 70
        },
        "monitoring_evidence": {
            "prometheus_endpoint": f"{PROMETHEUS_URL}",
            "metrics_endpoint": f"{BACKEND_URL}/metrics",
            "grafana_dashboard": "http://localhost:3001/d/lens-monitoring"
        }
    }
    
    # Save report
    with open("availability_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAvailability report saved to availability_report.json")
    
    # Example Prometheus queries for documentation
    print("\n" + "="*40)
    print("PROMETHEUS QUERIES FOR AVAILABILITY:")
    print("="*40)
    
    queries = [
        {
            "name": "Total Request Rate",
            "query": "sum(rate(http_requests_total[5m]))"
        },
        {
            "name": "Error Rate",
            "query": "sum(rate(http_requests_total{status=~'5..'}[5m]))"
        },
        {
            "name": "Availability Percentage",
            "query": "(1 - (sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m])))) * 100"
        },
        {
            "name": "7-day Availability",
            "query": "(1 - (sum(increase(http_requests_total{status=~'5..'}[7d])) / sum(increase(http_requests_total[7d])))) * 100"
        }
    ]
    
    for q in queries:
        print(f"\n{q['name']}:")
        print(f"  {q['query']}")
    
    return report

if __name__ == "__main__":
    calculate_availability_from_metrics()