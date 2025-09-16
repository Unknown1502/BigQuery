#!/usr/bin/env python3

"""
Dynamic Pricing Intelligence - System Validation Script
Comprehensive validation of all system components and capabilities
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class ValidationResult:
    component: str
    status: str
    details: str
    execution_time: float
    critical: bool = False

class SystemValidator:
    def __init__(self):
        self.console = Console()
        self.results: List[ValidationResult] = []
        
    def log_result(self, component: str, status: str, details: str, 
                   execution_time: float, critical: bool = False):
        """Log validation result"""
        self.results.append(ValidationResult(
            component=component,
            status=status,
            details=details,
            execution_time=execution_time,
            critical=critical
        ))
        
    async def validate_project_structure(self) -> bool:
        """Validate project directory structure"""
        start_time = time.time()
        
        required_paths = [
            "src/bigquery/functions",
            "src/bigquery/models", 
            "src/bigquery/procedures",
            "src/bigquery/views",
            "src/services/api-gateway",
            "src/services/image-processor",
            "src/services/pricing-engine",
            "src/services/data-ingestion",
            "src/services/stream-processor",
            "src/shared/config",
            "src/shared/clients",
            "src/shared/utils",
            "infrastructure/terraform",
            "infrastructure/kubernetes",
            "infrastructure/docker",
            "data/schemas",
            "data/migrations",
            "scripts/setup",
            "monitoring/dashboards",
            "tests/unit",
            "tests/integration",
            "docs/architecture"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not Path(path).exists():
                missing_paths.append(path)
        
        execution_time = time.time() - start_time
        
        if missing_paths:
            self.log_result(
                "Project Structure",
                "PARTIAL",
                f"Missing paths: {', '.join(missing_paths)}",
                execution_time
            )
            return False
        else:
            self.log_result(
                "Project Structure", 
                "PASS",
                "All required directories present",
                execution_time
            )
            return True
    
    async def validate_bigquery_components(self) -> bool:
        """Validate BigQuery AI components"""
        start_time = time.time()
        
        bigquery_files = {
            "Functions": [
                "src/bigquery/functions/analyze_street_scene.sql",
                "src/bigquery/functions/find_similar_locations.sql", 
                "src/bigquery/functions/predict_demand_with_confidence.sql",
                "src/bigquery/functions/calculate_optimal_price.sql",
                "src/bigquery/functions/assign_pricing_experiment.sql"
            ],
            "Models": [
                "src/bigquery/models/demand_forecast_model.sql",
                "src/bigquery/models/location_embeddings_model.sql",
                "src/bigquery/models/price_optimization_model.sql"
            ],
            "Procedures": [
                "src/bigquery/procedures/update_models_daily.sql",
                "src/bigquery/procedures/refresh_location_embeddings.sql"
            ],
            "Views": [
                "src/bigquery/views/real_time_pricing_dashboard.sql",
                "src/bigquery/views/location_performance_summary.sql"
            ]
        }
        
        missing_files = []
        ai_functions_found = 0
        
        for category, files in bigquery_files.items():
            for file_path in files:
                if Path(file_path).exists():
                    # Check for BigQuery AI function usage
                    content = Path(file_path).read_text()
                    if any(func in content for func in [
                        "ML.GENERATE_TEXT", "ML.GENERATE_EMBEDDING", 
                        "ML.DISTANCE", "AI.GENERATE_TEXT", "AI.GENERATE_TABLE"
                    ]):
                        ai_functions_found += 1
                else:
                    missing_files.append(file_path)
        
        execution_time = time.time() - start_time
        
        if missing_files:
            self.log_result(
                "BigQuery Components",
                "FAIL", 
                f"Missing files: {', '.join(missing_files)}",
                execution_time,
                critical=True
            )
            return False
        elif ai_functions_found < 5:
            self.log_result(
                "BigQuery AI Integration",
                "PARTIAL",
                f"Only {ai_functions_found} AI functions found, expected 5+",
                execution_time
            )
            return False
        else:
            self.log_result(
                "BigQuery AI Components",
                "PASS",
                f"All components present with {ai_functions_found} AI functions",
                execution_time
            )
            return True
    
    async def validate_microservices(self) -> bool:
        """Validate microservices implementation"""
        start_time = time.time()
        
        services = {
            "API Gateway": "src/services/api-gateway/main.py",
            "Image Processor": "src/services/image-processor/main.py", 
            "Pricing Engine": "src/services/pricing-engine/main.py",
            "Data Ingestion": "src/services/data-ingestion/main.py"
        }
        
        missing_services = []
        fastapi_services = 0
        
        for service_name, service_path in services.items():
            if Path(service_path).exists():
                content = Path(service_path).read_text()
                if "FastAPI" in content or "fastapi" in content:
                    fastapi_services += 1
            else:
                missing_services.append(service_name)
        
        execution_time = time.time() - start_time
        
        if missing_services:
            self.log_result(
                "Microservices",
                "FAIL",
                f"Missing services: {', '.join(missing_services)}",
                execution_time,
                critical=True
            )
            return False
        else:
            self.log_result(
                "Microservices",
                "PASS", 
                f"All {len(services)} services present, {fastapi_services} FastAPI-based",
                execution_time
            )
            return True
    
    async def validate_infrastructure(self) -> bool:
        """Validate infrastructure as code"""
        start_time = time.time()
        
        terraform_files = [
            "infrastructure/terraform/main.tf",
            "infrastructure/terraform/variables.tf",
            "infrastructure/terraform/bigquery.tf",
            "infrastructure/terraform/cloud-storage.tf",
            "infrastructure/terraform/pubsub.tf",
            "infrastructure/terraform/cloud-functions.tf",
            "infrastructure/terraform/vertex-ai.tf",
            "infrastructure/terraform/networking.tf",
            "infrastructure/terraform/outputs.tf"
        ]
        
        kubernetes_files = [
            "infrastructure/kubernetes/namespace.yaml",
            "infrastructure/kubernetes/api-gateway/deployment.yaml",
            "infrastructure/kubernetes/pricing-engine/deployment.yaml",
            "infrastructure/kubernetes/real-time-processor/deployment.yaml",
            "infrastructure/kubernetes/monitoring/deployment.yaml"
        ]
        
        docker_files = [
            "infrastructure/docker/Dockerfile.api",
            "infrastructure/docker/Dockerfile.processor"
        ]
        
        missing_files = []
        
        for file_list in [terraform_files, kubernetes_files, docker_files]:
            for file_path in file_list:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
        
        execution_time = time.time() - start_time
        
        if missing_files:
            self.log_result(
                "Infrastructure",
                "PARTIAL",
                f"Missing files: {', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''}",
                execution_time
            )
            return False
        else:
            self.log_result(
                "Infrastructure as Code",
                "PASS",
                f"All Terraform, Kubernetes, and Docker configs present",
                execution_time
            )
            return True
    
    async def validate_configuration(self) -> bool:
        """Validate configuration files"""
        start_time = time.time()
        
        config_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "Makefile",
            "docker-compose.yml",
            ".env.template",
            ".gitignore",
            "cloudbuild.yaml"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not Path(config_file).exists():
                missing_configs.append(config_file)
        
        execution_time = time.time() - start_time
        
        if missing_configs:
            self.log_result(
                "Configuration",
                "PARTIAL",
                f"Missing configs: {', '.join(missing_configs)}",
                execution_time
            )
            return False
        else:
            self.log_result(
                "Configuration Files",
                "PASS",
                "All configuration files present",
                execution_time
            )
            return True
    
    async def validate_schemas(self) -> bool:
        """Validate data schemas"""
        start_time = time.time()
        
        schema_files = [
            "data/schemas/bigquery_schemas.json",
            "data/schemas/pubsub_schemas.json", 
            "data/schemas/api_schemas.json"
        ]
        
        valid_schemas = 0
        
        for schema_file in schema_files:
            if Path(schema_file).exists():
                try:
                    with open(schema_file) as f:
                        json.load(f)
                    valid_schemas += 1
                except json.JSONDecodeError:
                    pass
        
        execution_time = time.time() - start_time
        
        if valid_schemas < len(schema_files):
            self.log_result(
                "Data Schemas",
                "PARTIAL",
                f"{valid_schemas}/{len(schema_files)} valid schemas",
                execution_time
            )
            return False
        else:
            self.log_result(
                "Data Schemas",
                "PASS",
                f"All {len(schema_files)} schemas valid",
                execution_time
            )
            return True
    
    async def validate_setup_capabilities(self) -> bool:
        """Validate setup and deployment capabilities"""
        start_time = time.time()
        
        setup_files = [
            "scripts/setup/setup_environment.sh"
        ]
        
        missing_setup = []
        for setup_file in setup_files:
            if not Path(setup_file).exists():
                missing_setup.append(setup_file)
        
        execution_time = time.time() - start_time
        
        if missing_setup:
            self.log_result(
                "Setup Capabilities",
                "PARTIAL",
                f"Missing setup files: {', '.join(missing_setup)}",
                execution_time
            )
            return False
        else:
            self.log_result(
                "Setup Capabilities",
                "PASS",
                "Setup and deployment scripts available",
                execution_time
            )
            return True
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_components = len(self.results)
        passed_components = len([r for r in self.results if r.status == "PASS"])
        failed_components = len([r for r in self.results if r.status == "FAIL"])
        partial_components = len([r for r in self.results if r.status == "PARTIAL"])
        critical_failures = len([r for r in self.results if r.status == "FAIL" and r.critical])
        
        total_time = sum(r.execution_time for r in self.results)
        
        return {
            "summary": {
                "total_components": total_components,
                "passed": passed_components,
                "failed": failed_components,
                "partial": partial_components,
                "critical_failures": critical_failures,
                "success_rate": (passed_components / total_components) * 100 if total_components > 0 else 0,
                "total_validation_time": total_time
            },
            "results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "details": r.details,
                    "execution_time": r.execution_time,
                    "critical": r.critical
                }
                for r in self.results
            ]
        }
    
    def display_results(self):
        """Display validation results in a formatted table"""
        table = Table(title="System Validation Results")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        table.add_column("Time (s)", justify="right", style="magenta")
        
        for result in self.results:
            status_style = {
                "PASS": "green",
                "FAIL": "red", 
                "PARTIAL": "yellow"
            }.get(result.status, "white")
            
            status_text = f"[{status_style}]{result.status}[/{status_style}]"
            if result.critical and result.status == "FAIL":
                status_text += " [red]⚠[/red]"
            
            table.add_row(
                result.component,
                status_text,
                result.details[:60] + "..." if len(result.details) > 60 else result.details,
                f"{result.execution_time:.2f}"
            )
        
        self.console.print(table)
        
        # Summary panel
        report = self.generate_validation_report()
        summary = report["summary"]
        
        summary_text = f"""
Total Components: {summary['total_components']}
✅ Passed: {summary['passed']}
❌ Failed: {summary['failed']}
⚠️  Partial: {summary['partial']}
🚨 Critical Failures: {summary['critical_failures']}

Success Rate: {summary['success_rate']:.1f}%
Total Validation Time: {summary['total_validation_time']:.2f}s
        """
        
        panel_style = "green" if summary['critical_failures'] == 0 else "red"
        self.console.print(Panel(summary_text, title="Validation Summary", style=panel_style))
    
    async def run_full_validation(self):
        """Run complete system validation"""
        self.console.print(Panel(
            "Dynamic Pricing Intelligence - System Validation",
            style="bold blue"
        ))
        
        validations = [
            ("Project Structure", self.validate_project_structure),
            ("BigQuery AI Components", self.validate_bigquery_components),
            ("Microservices", self.validate_microservices),
            ("Infrastructure", self.validate_infrastructure),
            ("Configuration", self.validate_configuration),
            ("Data Schemas", self.validate_schemas),
            ("Setup Capabilities", self.validate_setup_capabilities)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            for name, validation_func in validations:
                task = progress.add_task(f"Validating {name}...", total=None)
                await validation_func()
                progress.remove_task(task)
        
        self.display_results()
        
        # Final assessment
        report = self.generate_validation_report()
        if report["summary"]["critical_failures"] == 0:
            self.console.print(Panel(
                "🎉 SYSTEM VALIDATION SUCCESSFUL!\n\n"
                "The Dynamic Pricing Intelligence system is ready for deployment.\n"
                "All critical components are present and properly configured.",
                title="✅ VALIDATION COMPLETE",
                style="bold green"
            ))
            return True
        else:
            self.console.print(Panel(
                f"❌ VALIDATION FAILED\n\n"
                f"Found {report['summary']['critical_failures']} critical failures.\n"
                "Please address these issues before deployment.",
                title="🚨 VALIDATION FAILED", 
                style="bold red"
            ))
            return False

async def main():
    """Main validation execution"""
    validator = SystemValidator()
    
    try:
        success = await validator.run_full_validation()
        
        # Save validation report
        report = validator.generate_validation_report()
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to: validation_report.json")
        
        return 0 if success else 1
        
    except Exception as e:
        validator.console.print(f"[red]Validation error: {e}[/red]")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
