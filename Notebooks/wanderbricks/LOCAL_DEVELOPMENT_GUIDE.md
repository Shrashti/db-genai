# Local Development: Accessing Unity Catalog

## Fast Setup with `uv` (Recommended)
This project uses `uv` for fast dependency management.

1.  **Create Virtual Environment**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    uv pip install langchain mlflow databricks-sdk databricks-sql-connector pandas ipykernel langchain-community
    ```

## Option 1: Databricks SQL Connector (Recommended for this Agent)
Since our agent primarily executes SQL queries, the lightweight `databricks-sql-connector` is the easiest way to bridge your local environment to Databricks.

### 1. Install the Library
```bash
pip install databricks-sql-connector pandas
```

### 2. Set Environment Variables
You need your Databricks Workspace URL, an Access Token, and the HTTP Path of a SQL Warehouse.
```bash
export DATABRICKS_HOST="https://<your-workspace-instance-id>.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/..."
```

### 3. Update Code (Already done for you)
The `tools.py` has been updated to check for these variables. If found, it will create a connection and execute real SQL.

## Option 2: Databricks Connect
If you need full Spark DataFrame API capabilities (not just SQL), use Databricks Connect.

### 1. Install
```bash
pip install databricks-connect==13.3.0  # Match your cluster DBR version
```

### 2. Configure
```bash
databricks-connect configure
```

### 3. Use in Code
```python
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.getOrCreate()
df = spark.sql("SELECT * FROM wanderbricks.properties")
```

---
**Note**: The current `tools.py` implementation supports **Mock** (default), **Internal Spark** (Databricks Notebooks), and **Databricks SQL Connector** (Local Remote).
