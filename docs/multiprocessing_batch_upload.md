# Multiprocessing Batch Upload Feature

## Overview

The batch upload command now supports multiprocessing to significantly speed up the processing of multiple PDF files. By default, it uses 4 processors, but this can be customized by the user.

## New Command Line Options

### `--processors <number>`
- **Default**: 4 processors
- **Description**: Number of processors to use for parallel processing
- **Example**: `--processors 8` to use 8 processors

### `--sequential`
- **Description**: Force sequential processing (disable multiprocessing)
- **Use case**: When you need more detailed error messages or want to process files one by one

## Usage Examples

### Default multiprocessing (4 processors)
```bash
python -m src.core.cli batch-upload /path/to/papers/
```

### Custom number of processors
```bash
python -m src.core.cli batch-upload /path/to/papers/ --processors 8
```

### Sequential processing (original behavior)
```bash
python -m src.core.cli batch-upload /path/to/papers/ --sequential
```

### Force sequential with custom processors (processors will be ignored)
```bash
python -m src.core.cli batch-upload /path/to/papers/ --processors 8 --sequential
```

## Features

### Automatic Processor Validation
- Automatically detects available CPU cores
- Warns if requested processors exceed available cores
- Falls back to maximum available cores if needed

### Progress Tracking
- Real-time progress updates as files complete
- Success/failure indicators (✅/❌)
- Paper ID and basic stats for each processed file
- Final summary with success rate

### Error Handling
- Individual file failures don't stop the batch process
- Detailed error messages for failed files
- Suggestion to use `--sequential` for more detailed debugging

### Logging
- Comprehensive logging with START/FINISH messages
- Tracks each processing step
- Logs results for each file processed

## Performance Benefits

### Before (Sequential)
- Total time = sum of all individual processing times
- Uses only 1 CPU core
- No I/O parallelism

### After (Multiprocessing)
- Total time ≈ longest individual processing time (in ideal conditions)
- Utilizes multiple CPU cores
- Overlaps I/O operations

### Example Performance
- **10 PDF files, 2 minutes each**
  - Sequential: ~20 minutes
  - Multiprocessing (4 cores): ~5-6 minutes
  - **Speedup: ~3-4x faster**

## Technical Implementation

### Worker Function
- `process_single_pdf_worker()`: Module-level function for multiprocessing
- Each worker process initializes its own `DocumentProcessor`
- Returns structured results with success/error status

### Process Pool
- Uses `ProcessPoolExecutor` from `concurrent.futures`
- Spawn method for cross-platform compatibility
- Automatic cleanup of worker processes

### Progress Monitoring
- `as_completed()` iterator for real-time progress
- Non-blocking result collection
- Maintains order-independent processing

## Safety Features

### Resource Management
- Automatic cleanup of worker processes
- Memory isolation between processes
- Graceful handling of worker failures

### Error Isolation
- Individual file failures don't affect other files
- Comprehensive error reporting
- Fallback to sequential mode for debugging

## Logging Output

The system provides detailed logging for monitoring:

```
2024-01-15 10:30:00 - INFO - START: Batch upload command initiated
2024-01-15 10:30:00 - INFO - START: Searching for PDF files in /papers/
2024-01-15 10:30:01 - INFO - FINISH: Found 25 PDF files in /papers/
2024-01-15 10:30:01 - INFO - START: Parallel processing with 4 processors
2024-01-15 10:30:02 - INFO - START: Processing PDF file /papers/paper1.pdf
2024-01-15 10:32:15 - INFO - FINISH: Successfully processed /papers/paper1.pdf - Paper ID: abc123, Sentences: 150, Citations: 45
...
2024-01-15 10:45:30 - INFO - FINISH: Parallel processing completed
2024-01-15 10:45:30 - INFO - FINISH: Batch upload command completed
```

## Best Practices

### For Large Batches
- Use multiprocessing for batches of 5+ files
- Monitor system resources during processing
- Consider using fewer processors if system becomes unresponsive

### For Debugging
- Use `--sequential` flag for detailed error messages
- Check logs for specific failure reasons
- Process problematic files individually

### For Optimal Performance
- Use processor count equal to available CPU cores
- Ensure sufficient RAM for multiple DocumentProcessor instances
- Monitor disk I/O if processing many large files 