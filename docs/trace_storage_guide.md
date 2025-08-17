# Query Trace Storage System Guide

## Overview

The Query Trace Storage System provides persistent storage for tracking the execution of queries through the multi-agent system. It stores detailed information about each query, including execution steps, performance metrics, and final results.

## Features

- **Persistent Storage**: SQLite database for reliable data storage
- **Query Granularity**: Each query gets its own trace with unique ID
- **Performance Metrics**: Execution time and memory usage tracking
- **Error Handling**: Comprehensive error logging and recovery
- **Session Management**: Group queries by thread/session
- **Performance Analytics**: Agent-level performance statistics
- **Automatic Cleanup**: Configurable cleanup of old traces

## Database Schema

### Tables

#### 1. `queries` - Main query information
- `query_id`: Unique identifier for each query
- `query_text`: Original user query
- `thread_id`: Session/thread identifier
- `user_id`: User identifier
- `query_type`: Type of query (e.g., citation_analysis)
- `language`: User's language
- `created_at`: Query start time
- `completed_at`: Query completion time
- `total_steps`: Number of execution steps
- `execution_status`: Current status (running/completed)

#### 2. `query_steps` - Individual execution steps
- `step_id`: Unique step identifier
- `query_id`: Reference to parent query
- `agent_name`: Name of the agent that executed
- `step_number`: Sequential step number
- `input_data`: Input data for the step (JSON)
- `output_data`: Output data from the step (JSON)
- `execution_time`: Time taken for this step
- `memory_usage`: Memory used by this step
- `error_message`: Error message if any
- `created_at`: Step execution timestamp

#### 3. `query_results` - Final query results
- `result_id`: Unique result identifier
- `query_id`: Reference to parent query
- `final_response`: Final response text
- `confidence_score`: Confidence in the response
- `total_execution_time`: Total time for entire query
- `total_memory_usage`: Total memory used
- `data_sources_used`: Data sources accessed (JSON)
- `errors`: List of errors encountered (JSON)
- `warnings`: List of warnings (JSON)
- `created_at`: Result creation timestamp

## Usage

### Basic Usage

```python
from src.agents.multi_agent_system import LangGraphResearchSystem

# Initialize the system
system = LangGraphResearchSystem(
    graph_db=graph_db,
    vector_indexer=vector_indexer,
    author_index=author_index
)

# Process a query (tracing happens automatically)
result = await system.query(
    user_query="Which papers cite Porter's theory?",
    thread_id="session_001",
    user_id="user_123"
)
```

### Accessing Trace Data

```python
# Get complete trace for a specific query
trace = system.get_query_trace("abc123def4567890")
print(f"Query: {trace['query_info']['query_text']}")
print(f"Steps: {len(trace['steps'])}")
print(f"Final response: {trace['result']['final_response']}")

# Get all queries in a session
session_queries = system.get_session_queries("session_001", limit=10)
for query in session_queries:
    print(f"Query: {query['query_text']}")
    print(f"Status: {query['execution_status']}")

# Get performance statistics for an agent
agent_stats = system.get_agent_performance_stats("_language_processor_agent", days=7)
print(f"Average execution time: {agent_stats['avg_execution_time']:.3f}s")
print(f"Total executions: {agent_stats['total_executions']}")

# Get overall database statistics
db_stats = system.get_database_stats()
print(f"Total queries: {db_stats['total_queries']}")
print(f"Database size: {db_stats['database_size_mb']} MB")
```

### Manual Storage Management

```python
from src.storage.query_trace_storage import QueryTraceStorage

# Initialize storage
storage = QueryTraceStorage("logs/custom_traces.db")

# Start a trace
query_id = storage.start_query_trace(
    query_text="Custom query",
    thread_id="custom_thread",
    user_id="custom_user",
    query_type="custom",
    language="en"
)

# Add steps
storage.add_query_step(
    query_id=query_id,
    agent_name="custom_agent",
    step_number=1,
    input_data={"input": "data"},
    output_data={"output": "result"},
    execution_time=0.5,
    memory_usage=1024
)

# Complete the trace
storage.complete_query_trace(
    query_id=query_id,
    final_response="Custom response",
    confidence_score=0.9,
    total_execution_time=0.5,
    total_memory_usage=1024
)
```

## Configuration

### Database Path

The default database path is `logs/query_traces.db`. You can customize this:

```python
# In multi_agent_system.py
self.trace_storage = QueryTraceStorage("custom/path/traces.db")

# Or when creating storage directly
storage = QueryTraceStorage("custom/path/traces.db")
```

### Cleanup Settings

```python
# Clean up traces older than 30 days (default)
system.cleanup_old_traces(days=30)

# Clean up traces older than 7 days
system.cleanup_old_traces(days=7)
```

## Performance Considerations

### Memory Usage

- Each step stores input/output data as JSON strings
- Large responses are automatically truncated to prevent memory issues
- Consider regular cleanup for long-running systems

### Database Performance

- Indexes are created automatically for common queries
- Foreign key constraints ensure data integrity
- Consider database size monitoring for production use

### Monitoring

```python
# Regular monitoring example
import schedule
import time

def daily_maintenance():
    system.cleanup_old_traces(days=30)
    stats = system.get_database_stats()
    print(f"Daily maintenance completed. DB size: {stats['database_size_mb']} MB")

# Schedule daily cleanup at 2 AM
schedule.every().day.at("02:00").do(daily_maintenance)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Error Handling

The system includes comprehensive error handling:

- Database connection failures are logged and handled gracefully
- Invalid data is filtered out before storage
- Failed operations don't interrupt the main query flow
- All errors are logged with context information

## Migration from Old System

The new system maintains backward compatibility:

- Existing `export_trace_log()` method still works
- In-memory `agent_trace_log` is preserved
- New methods are additive, not replacing

## Troubleshooting

### Common Issues

1. **Database not created**: Check write permissions for logs directory
2. **Performance issues**: Monitor database size and cleanup old traces
3. **Memory errors**: Check for very large input/output data

### Debug Mode

```python
import logging
logging.getLogger('src.storage.query_trace_storage').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Compression**: Automatic compression of old trace data
- **Export Formats**: Support for CSV, JSON export
- **Real-time Monitoring**: Web dashboard for live trace monitoring
- **Advanced Analytics**: Machine learning insights from trace data
- **Distributed Storage**: Support for multiple storage backends

## API Reference

### QueryTraceStorage Methods

- `start_query_trace()`: Begin tracing a new query
- `add_query_step()`: Add execution step
- `complete_query_trace()`: Mark query as complete
- `get_query_trace()`: Retrieve complete trace
- `get_session_queries()`: Get queries in a session
- `get_agent_performance_stats()`: Get agent statistics
- `cleanup_old_traces()`: Remove old traces
- `get_database_stats()`: Get database statistics

### LangGraphResearchSystem Methods

- `get_query_trace(query_id)`: Get trace by ID
- `get_session_queries(thread_id, limit)`: Get session queries
- `get_agent_performance_stats(agent_name, days)`: Get agent stats
- `get_database_stats()`: Get overall stats
- `cleanup_old_traces(days)`: Cleanup old data
