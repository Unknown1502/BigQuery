#!/usr/bin/env python3
"""
API Endpoint Testing Script for Dynamic Pricing Intelligence System
Tests all API endpoints to ensure they return proper data
"""

import requests
import json
import time
import sys
from datetime import datetime

class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = {}
        
    def test_endpoint(self, endpoint, method="GET", data=None, expected_keys=None):
        """Test a single API endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            print(f"Testing {method} {endpoint}...")
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check status code
            if response.status_code != 200:
                self.results[endpoint] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text[:200]
                }
                print(f"  ❌ FAIL - HTTP {response.status_code}")
                return False
            
            # Parse JSON
            try:
                json_data = response.json()
            except json.JSONDecodeError:
                self.results[endpoint] = {
                    "status": "FAIL",
                    "error": "Invalid JSON response",
                    "response": response.text[:200]
                }
                print(f"  ❌ FAIL - Invalid JSON")
                return False
            
            # Check expected keys
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in json_data]
                if missing_keys:
                    self.results[endpoint] = {
                        "status": "FAIL",
                        "error": f"Missing keys: {missing_keys}",
                        "response": json_data
                    }
                    print(f"  ❌ FAIL - Missing keys: {missing_keys}")
                    return False
            
            self.results[endpoint] = {
                "status": "PASS",
                "response_size": len(response.text),
                "data_keys": list(json_data.keys()) if isinstance(json_data, dict) else "array"
            }
            print(f"  ✅ PASS - {len(response.text)} bytes")
            return True
            
        except requests.exceptions.RequestException as e:
            self.results[endpoint] = {
                "status": "FAIL",
                "error": f"Request failed: {str(e)}",
                "response": None
            }
            print(f"  ❌ FAIL - {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API endpoint tests"""
        print("🚀 Starting API Endpoint Tests")
        print("=" * 50)
        
        # Test cases
        test_cases = [
            {
                "endpoint": "/api/health",
                "method": "GET",
                "expected_keys": ["status", "timestamp", "service"]
            },
            {
                "endpoint": "/api/metrics",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/realtime-pricing",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/forecasts",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/forecasts?period=24h&location=all",
                "method": "GET",
                "expected_keys": ["success", "data", "period", "location"]
            },
            {
                "endpoint": "/api/clusters",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/features/importance",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/performance",
                "method": "GET",
                "expected_keys": ["success", "data"]
            },
            {
                "endpoint": "/api/performance?timeRange=1h",
                "method": "GET",
                "expected_keys": ["success", "data", "timeRange"]
            },
            {
                "endpoint": "/api/calculate-price",
                "method": "POST",
                "data": {
                    "latitude": "40.7128",
                    "longitude": "-74.0060",
                    "time_of_day": "12",
                    "day_of_week": "1",
                    "weather_condition": "clear",
                    "traffic_level": "medium",
                    "event_nearby": False,
                    "demand_multiplier": "1.0"
                },
                "expected_keys": ["success", "data"]
            }
        ]
        
        # Run tests
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            success = self.test_endpoint(
                test_case["endpoint"],
                test_case["method"],
                test_case.get("data"),
                test_case.get("expected_keys")
            )
            if success:
                passed += 1
            time.sleep(0.5)  # Small delay between tests
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the details above.")
            self.print_detailed_results()
            return False
    
    def print_detailed_results(self):
        """Print detailed test results"""
        print("\n📋 Detailed Results:")
        print("-" * 30)
        
        for endpoint, result in self.results.items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_icon} {endpoint}: {result['status']}")
            
            if result["status"] == "FAIL":
                print(f"   Error: {result['error']}")
                if result.get("response"):
                    print(f"   Response: {str(result['response'])[:100]}...")
            else:
                print(f"   Size: {result['response_size']} bytes")
                print(f"   Keys: {result['data_keys']}")
            print()

def main():
    """Main function"""
    print("Dynamic Pricing Intelligence System - API Testing")
    print("=" * 60)
    
    # Check if server is running
    tester = APITester()
    
    try:
        response = requests.get(f"{tester.base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Server is not running or not accessible")
        print("Please start the server first:")
        print("  cd ui && node server.js")
        sys.exit(1)
    
    # Run tests
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready to test the dashboard!")
        print("Open http://localhost:3000 in your browser")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the failing endpoints before testing the dashboard")
        sys.exit(1)

if __name__ == "__main__":
    main()
