#!/usr/bin/env python3
"""
Final Validation Script for BigQuery AI Competition Submission
Validates all components of the Dynamic Pricing Intelligence System
"""

import os
import subprocess
import sys
import json
from datetime import datetime

class FinalValidator:
    def __init__(self):
        self.results = {
            'terraform_validation': False,
            'bigquery_ai_features': False,
            'documentation_complete': False,
            'demo_functional': False,
            'competition_ready': False
        }
        self.errors = []
        
    def validate_terraform(self):
        """Validate Terraform configuration"""
        print("Validating Terraform configuration...")
        
        try:
            # Change to terraform directory
            terraform_dir = "infrastructure/terraform"
            if not os.path.exists(terraform_dir):
                self.errors.append("Terraform directory not found")
                return False
            
            # Run terraform validate
            result = subprocess.run(
                ["terraform", "validate"],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("SUCCESS: Terraform validation passed")
                self.results['terraform_validation'] = True
                return True
            else:
                print(f"ERROR: Terraform validation failed: {result.stderr}")
                self.errors.append(f"Terraform validation error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("ERROR: Terraform not found. Please install Terraform.")
            self.errors.append("Terraform not installed")
            return False
        except Exception as e:
            print(f"ERROR: Terraform validation error: {e}")
            self.errors.append(f"Terraform validation exception: {e}")
            return False
    
    def validate_bigquery_ai_features(self):
        """Validate BigQuery AI model files"""
        print("Validating BigQuery AI features...")
        
        required_models = [
            "src/bigquery/models/demand_forecast_model.sql",
            "src/bigquery/models/location_embeddings_model.sql", 
            "src/bigquery/models/price_optimization_model.sql",
            "src/bigquery/models/advanced_pricing_ai_model.sql"
        ]
        
        required_functions = [
            "src/bigquery/functions/analyze_street_scene.sql",
            "src/bigquery/functions/find_similar_locations.sql",
            "src/bigquery/functions/predict_demand_with_confidence.sql",
            "src/bigquery/functions/calculate_optimal_price.sql"
        ]
        
        all_files_exist = True
        
        # Check model files
        for model_file in required_models:
            if os.path.exists(model_file):
                print(f"FOUND: {model_file}")
            else:
                print(f"MISSING: {model_file}")
                self.errors.append(f"Missing BigQuery model: {model_file}")
                all_files_exist = False
        
        # Check function files
        for function_file in required_functions:
            if os.path.exists(function_file):
                print(f"FOUND: {function_file}")
            else:
                print(f"MISSING: {function_file}")
                self.errors.append(f"Missing BigQuery function: {function_file}")
                all_files_exist = False
        
        # Check for BigQuery AI features in model files
        bigquery_ai_features = [
            "ARIMA_PLUS",
            "ML.GENERATE_EMBEDDING", 
            "ML.PREDICT",
            "ML.EVALUATE",
            "KMEANS"
        ]
        
        features_found = set()
        
        for model_file in required_models:
            if os.path.exists(model_file):
                with open(model_file, 'r') as f:
                    content = f.read().upper()
                    for feature in bigquery_ai_features:
                        if feature in content:
                            features_found.add(feature)
        
        print(f"BigQuery AI features found: {', '.join(features_found)}")
        
        if len(features_found) >= 3 and all_files_exist:
            print("SUCCESS: BigQuery AI features validation passed")
            self.results['bigquery_ai_features'] = True
            return True
        else:
            print("ERROR: Insufficient BigQuery AI features")
            self.errors.append("Missing required BigQuery AI features")
            return False
    
    def validate_documentation(self):
        """Validate documentation completeness"""
        print("Validating documentation...")
        
        required_docs = [
            "README.md",
            "docs/ARCHITECTURE.md",
            "docs/BLOG_POST.md", 
            "docs/BIGQUERY_AI_FEEDBACK.md"
        ]
        
        all_docs_exist = True
        
        for doc_file in required_docs:
            if os.path.exists(doc_file):
                # Check file size to ensure it's not empty
                file_size = os.path.getsize(doc_file)
                if file_size > 1000:  # At least 1KB of content
                    print(f"FOUND: {doc_file} ({file_size} bytes)")
                else:
                    print(f"ERROR: {doc_file} is too small ({file_size} bytes)")
                    self.errors.append(f"Documentation file too small: {doc_file}")
                    all_docs_exist = False
            else:
                print(f"MISSING: {doc_file}")
                self.errors.append(f"Missing documentation: {doc_file}")
                all_docs_exist = False
        
        if all_docs_exist:
            print("SUCCESS: Documentation validation passed")
            self.results['documentation_complete'] = True
            return True
        else:
            print("ERROR: Documentation validation failed")
            return False
    
    def validate_demo_functionality(self):
        """Validate demo script functionality"""
        print("Validating demo functionality...")
        
        demo_script = "scripts/demo/bigquery_ai_demo.py"
        
        if not os.path.exists(demo_script):
            print(f"ERROR: Demo script not found: {demo_script}")
            self.errors.append("Demo script missing")
            return False
        
        # Check if demo script has required components
        with open(demo_script, 'r') as f:
            content = f.read()
            
        required_components = [
            "BigQueryAIDemo",
            "ARIMA_PLUS",
            "ML.GENERATE_EMBEDDING",
            "K-MEANS",
            "demonstrate_"
        ]
        
        components_found = []
        for component in required_components:
            if component in content:
                components_found.append(component)
        
        if len(components_found) >= 4:
            print(f"SUCCESS: Demo script contains required components: {', '.join(components_found)}")
            self.results['demo_functional'] = True
            return True
        else:
            print(f"ERROR: Demo script missing components. Found: {', '.join(components_found)}")
            self.errors.append("Demo script incomplete")
            return False
    
    def validate_competition_readiness(self):
        """Validate overall competition readiness"""
        print("Validating competition readiness...")
        
        # Check infrastructure files
        infrastructure_files = [
            "infrastructure/terraform/main.tf",
            "infrastructure/terraform/variables.tf",
            "infrastructure/terraform/bigquery.tf",
            "infrastructure/kubernetes/namespace.yaml"
        ]
        
        # Check service files
        service_files = [
            "src/services/api-gateway/main.py",
            "src/services/pricing-engine/main.py",
            "src/services/image-processor/main.py"
        ]
        
        all_files_present = True
        
        for file_path in infrastructure_files + service_files:
            if os.path.exists(file_path):
                print(f"FOUND: {file_path}")
            else:
                print(f"MISSING: {file_path}")
                self.errors.append(f"Missing file: {file_path}")
                all_files_present = False
        
        # Check if all other validations passed
        validations_passed = (
            self.results['terraform_validation'] and
            self.results['bigquery_ai_features'] and
            self.results['documentation_complete'] and
            self.results['demo_functional'] and
            all_files_present
        )
        
        if validations_passed:
            print("SUCCESS: Competition submission is ready")
            self.results['competition_ready'] = True
            return True
        else:
            print("ERROR: Competition submission not ready")
            return False
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("FINAL VALIDATION REPORT")
        print("=" * 60)
        
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Project: Dynamic Pricing Intelligence System")
        print(f"Competition: BigQuery AI")
        
        print("\nValidation Results:")
        for check, result in self.results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {check}: {status}")
        
        if self.errors:
            print("\nErrors Found:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        overall_status = "READY" if self.results['competition_ready'] else "NOT READY"
        print(f"\nOverall Status: {overall_status}")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'project': 'Dynamic Pricing Intelligence System',
            'competition': 'BigQuery AI',
            'results': self.results,
            'errors': self.errors,
            'overall_status': overall_status
        }
        
        with open('FINAL_VALIDATION_REPORT.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: FINAL_VALIDATION_REPORT.json")
        
        return self.results['competition_ready']
    
    def run_all_validations(self):
        """Run all validation checks"""
        print("Starting Final Validation for BigQuery AI Competition Submission")
        print("=" * 60)
        
        # Run all validation checks
        self.validate_terraform()
        self.validate_bigquery_ai_features()
        self.validate_documentation()
        self.validate_demo_functionality()
        self.validate_competition_readiness()
        
        # Generate final report
        return self.generate_report()

def main():
    """Main function"""
    validator = FinalValidator()
    success = validator.run_all_validations()
    
    if success:
        print("\nCongratulations! Your BigQuery AI competition submission is ready.")
        print("Next steps:")
        print("1. Push code to GitHub repository")
        print("2. Publish blog post")
        print("3. Submit to BigQuery AI competition")
        sys.exit(0)
    else:
        print("\nValidation failed. Please fix the errors above before submitting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
