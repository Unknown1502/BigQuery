# Error Resolution Plan

## SQL Syntax Errors (BigQuery)

### 1. competitor_analysis_view.sql
- **Issue**: PERCENTILE_CONT syntax error on line 31
- **Fix**: Remove OVER clause from PERCENTILE_CONT as it's an aggregate function

### 2. find_similar_locations.sql  
- **Issue**: Table-valued function syntax errors
- **Fix**: Correct the RETURNS TABLE syntax and function structure

### 3. real_time_pricing_dashboard.sql
- **Issues**: Multiple syntax errors with ML.FORECAST, STRUCT, and timestamp operations
- **Fix**: Correct ML.FORECAST syntax, fix STRUCT usage, and timestamp arithmetic

## Python Pylance Errors

### 4. gcp_config.py
- **Issue**: Settings() constructor called with 0 arguments but expects parameters
- **Fix**: Update settings import and usage

### 5. base_models.py & location_models.py
- **Issue**: None assigned to str parameters
- **Fix**: Add proper null checks and default values

### 6. error_handling_fixed.py
- **Issue**: None assigned to str parameters  
- **Fix**: Add null checks for string parameters

### 7. gcp_utils.py
- **Issue**: Accessing unknown attributes on Exception class
- **Fix**: Add proper exception type checking

### 8. monitoring_utils.py
- **Issue**: Constructor argument issues
- **Fix**: Update constructor calls

### 9. test_street_camera_datasets.py
- **Issue**: ModuleSpec | None type issues
- **Fix**: Add null checks for module loading

### 10. data-ingestion/main.py
- **Issue**: Missing config argument
- **Fix**: Add required config parameter

## Implementation Steps

1. ✅ Analyze all error files
2. ⏳ Fix SQL syntax errors in BigQuery files
3. ⏳ Fix Python type and import errors
4. ⏳ Test fixes and validate syntax
5. ⏳ Final validation of all fixes

## Priority Order
1. SQL files (blocking database operations)
2. Core shared modules (base_models, gcp_config)
3. Service-specific files
4. Test files
