"""
Monitoring Utilities - Advanced observability and performance tracking
Handles metrics collection, alerting, and system health monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings import get_settings
from src.shared.utils.logging_utils import setup_logging

# Setup logging
logger = setup_logging(__name__)
settings = get_settings()

# Prometheus metrics registry
REGISTRY = CollectorRegistry()

# Core metrics
REQUEST_COUNT = Counter(
    'pricing_requests_total',
    'Total number of pricing requests',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'pricing_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    registry=REGISTRY
)

PRICING_CALCULATION_DURATION = Histogram(
    'pricing_calculation_duration_seconds',
    'Time spent calculating optimal price',
    ['location_type', 'complexity'],
    registry=REGISTRY
)

IMAGE_PROCESSING_DURATION = Histogram(
    'image_processing_duration_seconds',
    'Time spent processing street imagery',
    ['processing_type', 'image_size'],
    registry=REGISTRY
)

BIGQUERY_QUERY_DURATION = Histogram(
    'bigquery_query_duration_seconds',
    'BigQuery query execution time',
    ['query_type', 'dataset'],
    registry=REGISTRY
)

SURGE_MULTIPLIER = Histogram(
    'surge_multiplier',
    'Current surge multiplier values',
    ['location_id', 'time_of_day'],
    buckets=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    registry=REGISTRY
)

DEMAND_PREDICTION_ACCURACY = Gauge(
    'demand_prediction_accuracy',
    'Accuracy of demand predictions',
    ['model_version', 'location_type'],
    registry=REGISTRY
)

SYSTEM_HEALTH = Gauge(
    'system_health_score',
    'Overall system health score (0-1)',
    ['component'],
    registry=REGISTRY
)

ML_MODEL_LATENCY = Histogram(
    'ml_model_latency_seconds',
    'ML model inference latency',
    ['model_name', 'model_version'],
    registry=REGISTRY
)

REVENUE_IMPACT = Counter(
    'revenue_impact_total',
    'Total revenue impact from pricing optimization',
    ['location_id', 'strategy'],
    registry=REGISTRY
)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    active_connections: int = 0
    request_rate: float = 0.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0

@dataclass
class BusinessMetrics:
    """Container for business metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_rides: int = 0
    revenue_per_ride: float = 0.0
    surge_events: int = 0
    customer_satisfaction: float = 0.0
    market_share: float = 0.0
    pricing_accuracy: float = 0.0

class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_counts = {}
        self.response_times = []
        self.error_counts = {}
        
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str = 'POST'):
        """Context manager for tracking request metrics"""
        start_time = time.time()
        status = 'success'
        
        try:
            yield
        except Exception as e:
            status = 'error'
            logger.error(f"Request failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            
            # Update Prometheus metrics
            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()
            
            REQUEST_DURATION.labels(
                endpoint=endpoint,
                method=method
            ).observe(duration)
            
            # Update internal tracking
            key = f"{endpoint}:{method}"
            self.request_counts[key] = self.request_counts.get(key, 0) + 1
            self.response_times.append(duration)
            
            if status == 'error':
                self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    @asynccontextmanager
    async def track_pricing_calculation(self, location_type: str = 'urban', complexity: str = 'standard'):
        """Track pricing calculation performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            PRICING_CALCULATION_DURATION.labels(
                location_type=location_type,
                complexity=complexity
            ).observe(duration)
    
    @asynccontextmanager
    async def track_image_processing(self, processing_type: str, image_size: str = 'medium'):
        """Track image processing performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            IMAGE_PROCESSING_DURATION.labels(
                processing_type=processing_type,
                image_size=image_size
            ).observe(duration)
    
    @asynccontextmanager
    async def track_bigquery_query(self, query_type: str, dataset: str = 'ride_intelligence'):
        """Track BigQuery query performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            BIGQUERY_QUERY_DURATION.labels(
                query_type=query_type,
                dataset=dataset
            ).observe(duration)
    
    @asynccontextmanager
    async def track_ml_inference(self, model_name: str, model_version: str = 'v1.0'):
        """Track ML model inference performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            ML_MODEL_LATENCY.labels(
                model_name=model_name,
                model_version=model_version
            ).observe(duration)
    
    def record_surge_multiplier(self, location_id: str, multiplier: float, time_of_day: str = 'unknown'):
        """Record surge multiplier value"""
        SURGE_MULTIPLIER.labels(
            location_id=location_id,
            time_of_day=time_of_day
        ).observe(multiplier)
    
    def update_demand_accuracy(self, accuracy: float, model_version: str = 'v1.0', location_type: str = 'urban'):
        """Update demand prediction accuracy"""
        DEMAND_PREDICTION_ACCURACY.labels(
            model_version=model_version,
            location_type=location_type
        ).set(accuracy)
    
    def update_system_health(self, component: str, health_score: float):
        """Update system health score"""
        SYSTEM_HEALTH.labels(component=component).set(health_score)
    
    def record_revenue_impact(self, location_id: str, revenue_impact: float, strategy: str = 'dynamic'):
        """Record revenue impact from pricing optimization"""
        REVENUE_IMPACT.labels(
            location_id=location_id,
            strategy=strategy
        ).inc(revenue_impact)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Application metrics
            uptime = time.time() - self.start_time
            total_requests = sum(self.request_counts.values())
            request_rate = total_requests / uptime if uptime > 0 else 0
            
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0
            )
            
            return PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                request_rate=request_rate,
                error_rate=error_rate,
                avg_response_time=avg_response_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return generate_latest(REGISTRY).decode('utf-8')

class AlertManager:
    """Advanced alerting and notification system"""
    
    def __init__(self):
        self.alert_rules = {}
        self.alert_history = []
        self.notification_channels = []
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                      severity: str = 'warning', cooldown: int = 300):
        """Add alert rule with condition function"""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'cooldown': cooldown,
            'last_triggered': None,
            'trigger_count': 0
        }
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, slack, webhook, etc.)"""
        self.notification_channels.append({
            'type': channel_type,
            'config': config,
            'enabled': True
        })
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        current_time = datetime.utcnow()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check cooldown period
                if (rule['last_triggered'] and 
                    (current_time - rule['last_triggered']).seconds < rule['cooldown']):
                    continue
                
                # Evaluate condition
                if rule['condition'](metrics):
                    await self._trigger_alert(rule_name, rule, metrics)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    async def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger alert and send notifications"""
        alert = {
            'rule_name': rule_name,
            'severity': rule['severity'],
            'timestamp': datetime.utcnow(),
            'metrics': metrics,
            'message': f"Alert triggered: {rule_name}"
        }
        
        # Update rule state
        rule['last_triggered'] = alert['timestamp']
        rule['trigger_count'] += 1
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.notification_channels:
            if channel['enabled']:
                await self._send_notification(channel, alert)
        
        logger.warning(f"Alert triggered: {rule_name} (severity: {rule['severity']})")
    
    async def _send_notification(self, channel: Dict[str, Any], alert: Dict[str, Any]):
        """Send notification through specified channel"""
        try:
            if channel['type'] == 'webhook':
                # Send webhook notification
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        channel['config']['url'],
                        json=alert,
                        headers=channel['config'].get('headers', {})
                    )
            
            elif channel['type'] == 'log':
                # Log notification
                logger.critical(f"ALERT: {alert['message']}")
            
            # Add more notification types as needed
            
        except Exception as e:
            logger.error(f"Failed to send notification via {channel['type']}: {e}")

class HealthChecker:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool],
                            timeout: int = 30, critical: bool = False):
        """Register a health check function"""
        self.health_checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'critical': critical,
            'last_result': None,
            'last_check': None
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        critical_failures = 0
        
        for check_name, check_config in self.health_checks.items():
            try:
                # Run health check with timeout
                start_time = time.time()
                
                check_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, check_config['func']
                    ),
                    timeout=check_config['timeout']
                )
                
                duration = time.time() - start_time
                
                results['checks'][check_name] = {
                    'status': 'healthy' if check_result else 'unhealthy',
                    'duration': duration,
                    'critical': check_config['critical'],
                    'last_check': datetime.utcnow().isoformat()
                }
                
                # Update check state
                check_config['last_result'] = check_result
                check_config['last_check'] = datetime.utcnow()
                
                # Count critical failures
                if not check_result and check_config['critical']:
                    critical_failures += 1
                
            except asyncio.TimeoutError:
                results['checks'][check_name] = {
                    'status': 'timeout',
                    'duration': check_config['timeout'],
                    'critical': check_config['critical'],
                    'error': 'Health check timed out'
                }
                
                if check_config['critical']:
                    critical_failures += 1
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'critical': check_config['critical'],
                    'error': str(e)
                }
                
                if check_config['critical']:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            results['overall_status'] = 'critical'
        elif any(check['status'] != 'healthy' for check in results['checks'].values()):
            results['overall_status'] = 'degraded'
        
        self.last_check_results = results
        return results

# Global instances
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
health_checker = HealthChecker()

# Default alert rules
def setup_default_alerts():
    """Setup default alert rules for the pricing system"""
    
    # High error rate alert
    alert_manager.add_alert_rule(
        'high_error_rate',
        lambda m: m.get('error_rate', 0) > 0.05,  # 5% error rate
        severity='critical',
        cooldown=300
    )
    
    # High response time alert
    alert_manager.add_alert_rule(
        'high_response_time',
        lambda m: m.get('avg_response_time', 0) > 1.0,  # 1 second
        severity='warning',
        cooldown=600
    )
    
    # High CPU usage alert
    alert_manager.add_alert_rule(
        'high_cpu_usage',
        lambda m: m.get('cpu_usage', 0) > 80,  # 80% CPU
        severity='warning',
        cooldown=300
    )
    
    # High memory usage alert
    alert_manager.add_alert_rule(
        'high_memory_usage',
        lambda m: m.get('memory_usage', 0) > 85,  # 85% memory
        severity='critical',
        cooldown=300
    )

# Default health checks
def setup_default_health_checks():
    """Setup default health checks"""
    
    def check_disk_space():
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('/')
            return (disk.free / disk.total) > 0.1  # At least 10% free
        except:
            return False
    
    def check_memory():
        """Check available memory"""
        try:
            memory = psutil.virtual_memory()
            return memory.available > (1024 * 1024 * 1024)  # At least 1GB available
        except:
            return False
    
    health_checker.register_health_check('disk_space', check_disk_space, critical=True)
    health_checker.register_health_check('memory', check_memory, critical=True)

# Initialize default configurations
setup_default_alerts()
setup_default_health_checks()
