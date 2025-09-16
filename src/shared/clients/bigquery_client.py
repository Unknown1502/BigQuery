"""
BigQuery client wrapper with advanced functionality for the Dynamic Pricing Intelligence System.
Provides high-level interfaces for BigQuery operations, streaming, and ML model interactions.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, LoadJobConfig, WriteDisposition
from google.cloud.bigquery.job import QueryJob
from google.cloud.exceptions import NotFound, BadRequest, Forbidden

from ..config.settings import settings
from ..utils.logging_utils import get_logger
from ..utils.error_handling import PricingIntelligenceError
from ..utils.gcp_utils import handle_gcp_error


logger = get_logger(__name__)


class BigQueryClientError(PricingIntelligenceError):
    """Custom exception for BigQuery client errors."""
    pass


class BigQueryClient:
    """
    Advanced BigQuery client wrapper with streaming, ML, and analytics capabilities.
    """
    
    def __init__(self):
        """Initialize BigQuery client with configuration."""
        self.client = bigquery.Client(
            project=settings.database.project_id,
            location=settings.database.location
        )
        self.project_id = settings.database.project_id
        self.dataset_id = settings.database.dataset_id
        self.dataset_ref = self.client.dataset(self.dataset_id)
        
        # Streaming insert buffer
        self._streaming_buffer: List[Dict[str, Any]] = []
        self._streaming_buffer_lock = asyncio.Lock()
        
        logger.info(
            "BigQuery client initialized",
            extra={
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "location": settings.database.location
            }
        )
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        use_cache: bool = True,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute a BigQuery SQL query with parameters and return results.
        
        Args:
            query: SQL query string
            parameters: Query parameters for parameterized queries
            timeout: Query timeout in seconds
            use_cache: Whether to use query cache
            dry_run: Whether to perform a dry run (validation only)
            
        Returns:
            List of query result rows as dictionaries
            
        Raises:
            BigQueryClientError: If query execution fails
        """
        try:
            job_config = QueryJobConfig(
                use_query_cache=use_cache,
                dry_run=dry_run,
                use_legacy_sql=False
            )
            
            # Add query parameters if provided
            if parameters:
                job_config.query_parameters = self._build_query_parameters(parameters)
            
            # Execute query
            query_job = self.client.query(
                query,
                job_config=job_config,
                timeout=timeout or settings.database.query_timeout_seconds
            )
            
            if dry_run:
                logger.info(
                    "Query validation successful",
                    extra={
                        "query_hash": hash(query),
                        "total_bytes_processed": query_job.total_bytes_processed
                    }
                )
                return []
            
            # Wait for query completion and get results
            results = query_job.result()
            rows = [dict(row) for row in results]
            
            logger.info(
                "Query executed successfully",
                extra={
                    "query_hash": hash(query),
                    "rows_returned": len(rows),
                    "total_bytes_processed": query_job.total_bytes_processed,
                    "slot_millis": query_job.slot_millis,
                    "job_id": query_job.job_id
                }
            )
            
            return rows
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg, extra={"query_hash": hash(query)})
            raise BigQueryClientError(error_msg) from e
    
    async def stream_insert(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        ignore_unknown_values: bool = False,
        skip_invalid_rows: bool = False
    ) -> None:
        """
        Stream insert rows into a BigQuery table.
        
        Args:
            table_name: Target table name
            rows: List of row dictionaries to insert
            ignore_unknown_values: Whether to ignore unknown fields
            skip_invalid_rows: Whether to skip invalid rows
            
        Raises:
            BigQueryClientError: If streaming insert fails
        """
        try:
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)
            
            # Add insertion timestamp to each row
            timestamp = datetime.now(timezone.utc).isoformat()
            for row in rows:
                if 'inserted_at' not in row:
                    row['inserted_at'] = timestamp
            
            # Perform streaming insert
            errors = self.client.insert_rows_json(
                table,
                rows,
                ignore_unknown_values=ignore_unknown_values,
                skip_invalid_rows=skip_invalid_rows
            )
            
            if errors:
                error_msg = f"Streaming insert errors: {errors}"
                logger.error(
                    error_msg,
                    extra={
                        "table_name": table_name,
                        "rows_attempted": len(rows),
                        "errors": errors
                    }
                )
                raise BigQueryClientError(error_msg)
            
            logger.info(
                "Streaming insert successful",
                extra={
                    "table_name": table_name,
                    "rows_inserted": len(rows)
                }
            )
            
        except Exception as e:
            error_msg = f"Streaming insert failed for table {table_name}: {str(e)}"
            logger.error(error_msg)
            raise BigQueryClientError(error_msg) from e
    
    async def buffered_stream_insert(
        self,
        table_name: str,
        row: Dict[str, Any],
        buffer_size: Optional[int] = None
    ) -> None:
        """
        Add row to streaming buffer and flush when buffer is full.
        
        Args:
            table_name: Target table name
            row: Row dictionary to insert
            buffer_size: Buffer size (uses default if not specified)
        """
        buffer_size = buffer_size or settings.database.streaming_buffer_size
        
        async with self._streaming_buffer_lock:
            # Add table name to row for routing
            row['_target_table'] = table_name
            self._streaming_buffer.append(row)
            
            # Flush buffer if full
            if len(self._streaming_buffer) >= buffer_size:
                await self._flush_streaming_buffer()
    
    async def flush_streaming_buffer(self) -> None:
        """Manually flush the streaming buffer."""
        async with self._streaming_buffer_lock:
            await self._flush_streaming_buffer()
    
    async def _flush_streaming_buffer(self) -> None:
        """Internal method to flush streaming buffer."""
        if not self._streaming_buffer:
            return
        
        # Group rows by target table
        table_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._streaming_buffer:
            table_name = row.pop('_target_table')
            if table_name not in table_rows:
                table_rows[table_name] = []
            table_rows[table_name].append(row)
        
        # Stream insert to each table
        for table_name, rows in table_rows.items():
            try:
                await self.stream_insert(table_name, rows)
            except Exception as e:
                logger.error(
                    f"Failed to flush buffer for table {table_name}: {str(e)}",
                    extra={"table_name": table_name, "rows_count": len(rows)}
                )
        
        # Clear buffer
        self._streaming_buffer.clear()
        
        logger.debug(
            "Streaming buffer flushed",
            extra={"tables_flushed": len(table_rows)}
        )
    
    async def call_bigquery_function(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Any:
        """
        Call a BigQuery user-defined function.
        
        Args:
            function_name: Name of the BigQuery function
            parameters: Function parameters
            timeout: Query timeout in seconds
            
        Returns:
            Function result
        """
        # Build function call query
        param_placeholders = ", ".join([f"@{key}" for key in parameters.keys()])
        query = f"SELECT `{self.dataset_id}.{function_name}`({param_placeholders}) AS result"
        
        results = await self.execute_query(query, parameters, timeout)
        
        if not results:
            raise BigQueryClientError(f"Function {function_name} returned no results")
        
        return results[0]['result']
    
    async def analyze_street_scene(
        self,
        image_ref: str,
        location_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the analyze_street_scene BigQuery AI function.
        
        Args:
            image_ref: Reference to the image in Cloud Storage
            location_data: Location context data
            
        Returns:
            Street scene analysis results
        """
        parameters = {
            "image_ref": image_ref,
            "location_data": location_data
        }
        
        return await self.call_bigquery_function("analyze_street_scene", parameters)
    
    async def find_similar_locations(
        self,
        target_location_id: str,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find locations similar to the target location using embeddings.
        
        Args:
            target_location_id: ID of the target location
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar locations with similarity scores
        """
        query = f"""
        SELECT * FROM `{self.dataset_id}.find_similar_locations`(@target_location_id, @similarity_threshold)
        """
        
        parameters = {
            "target_location_id": target_location_id,
            "similarity_threshold": similarity_threshold
        }
        
        return await self.execute_query(query, parameters)
    
    async def predict_demand_with_confidence(
        self,
        location_id: str,
        prediction_horizon_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Get demand prediction with confidence intervals.
        
        Args:
            location_id: Location identifier
            prediction_horizon_hours: Hours ahead to predict
            
        Returns:
            Demand prediction with confidence intervals
        """
        parameters = {
            "location_id": location_id,
            "prediction_horizon_hours": prediction_horizon_hours
        }
        
        return await self.call_bigquery_function("predict_demand_with_confidence", parameters)
    
    async def calculate_optimal_price(
        self,
        location_id: str,
        current_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal price using BigQuery AI pricing engine.
        
        Args:
            location_id: Location identifier
            current_timestamp: Current timestamp (uses now if not provided)
            
        Returns:
            Optimal pricing calculation results
        """
        if current_timestamp is None:
            current_timestamp = datetime.now(timezone.utc)
        
        parameters = {
            "location_id": location_id,
            "current_timestamp": current_timestamp.isoformat()
        }
        
        return await self.call_bigquery_function("calculate_optimal_price", parameters)
    
    async def assign_pricing_experiment(
        self,
        location_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Assign user to pricing experiment group.
        
        Args:
            location_id: Location identifier
            user_id: User identifier
            
        Returns:
            Experiment assignment details
        """
        parameters = {
            "location_id": location_id,
            "user_id": user_id
        }
        
        return await self.call_bigquery_function("assign_pricing_experiment", parameters)
    
    async def get_real_time_dashboard_data(
        self,
        location_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get real-time pricing dashboard data.
        
        Args:
            location_ids: Optional list of specific location IDs
            
        Returns:
            Dashboard data for locations
        """
        query = f"SELECT * FROM `{self.dataset_id}.real_time_pricing_dashboard`"
        
        if location_ids:
            placeholders = ", ".join([f"'{loc_id}'" for loc_id in location_ids])
            query += f" WHERE location_id IN ({placeholders})"
        
        query += " ORDER BY location_health_score DESC"
        
        return await self.execute_query(query)
    
    async def get_location_performance_summary(
        self,
        location_ids: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get location performance summary data.
        
        Args:
            location_ids: Optional list of specific location IDs
            min_score: Minimum overall location score filter
            
        Returns:
            Performance summary data for locations
        """
        query = f"SELECT * FROM `{self.dataset_id}.location_performance_summary`"
        
        conditions = []
        if location_ids:
            placeholders = ", ".join([f"'{loc_id}'" for loc_id in location_ids])
            conditions.append(f"location_id IN ({placeholders})")
        
        if min_score is not None:
            conditions.append(f"overall_location_score >= {min_score}")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY overall_location_score DESC"
        
        return await self.execute_query(query)
    
    def _build_query_parameters(self, parameters: Dict[str, Any]) -> List[bigquery.ScalarQueryParameter]:
        """Build BigQuery query parameters from dictionary."""
        query_parameters = []
        
        for key, value in parameters.items():
            if isinstance(value, str):
                param_type = "STRING"
            elif isinstance(value, int):
                param_type = "INT64"
            elif isinstance(value, float):
                param_type = "FLOAT64"
            elif isinstance(value, bool):
                param_type = "BOOL"
            elif isinstance(value, datetime):
                param_type = "TIMESTAMP"
                value = value.isoformat()
            elif isinstance(value, dict):
                param_type = "JSON"
                value = json.dumps(value)
            else:
                param_type = "STRING"
                value = str(value)
            
            query_parameters.append(
                bigquery.ScalarQueryParameter(key, param_type, value)
            )
        
        return query_parameters
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on BigQuery connection and dataset.
        
        Returns:
            Health check results
        """
        try:
            # Test basic connectivity
            query = "SELECT 1 as health_check"
            await self.execute_query(query, timeout=10)
            
            # Check dataset access
            dataset = self.client.get_dataset(self.dataset_ref)
            
            # Check table access
            tables = list(self.client.list_tables(dataset))
            
            return {
                "status": "healthy",
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "tables_count": len(tables),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Alias for execute_query for backward compatibility.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            List of query result rows as dictionaries
        """
        return await self.execute_query(query, parameters)
    
    async def query_async(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Async query method for backward compatibility.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            List of query result rows as dictionaries
        """
        return await self.execute_query(query, parameters)
    
    async def insert_rows(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        """
        Insert rows method for backward compatibility.
        
        Args:
            table_name: Target table name
            rows: List of row dictionaries to insert
        """
        await self.stream_insert(table_name, rows)

    async def close(self) -> None:
        """Clean up resources and flush any remaining buffered data."""
        await self.flush_streaming_buffer()
        logger.info("BigQuery client closed")


# Global BigQuery client instance
_bigquery_client: Optional[BigQueryClient] = None


def get_bigquery_client() -> BigQueryClient:
    """Get or create global BigQuery client instance."""
    global _bigquery_client
    
    if _bigquery_client is None:
        _bigquery_client = BigQueryClient()
    
    return _bigquery_client


@asynccontextmanager
async def bigquery_client_context():
    """Context manager for BigQuery client with automatic cleanup."""
    client = get_bigquery_client()
    try:
        yield client
    finally:
        await client.close()
