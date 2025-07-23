# Resume Batch Upload Feature

## Overview

The batch upload system now supports resuming interrupted uploads by tracking processing progress in a local JSON file. This allows you to continue from where you left off if the process is interrupted by errors, system shutdown, or other issues.

## How It Works

### Progress Tracking
- Progress is automatically tracked in `data/batch_upload_tracker.json`
- Each file's status is recorded: `completed`, `failed`, or `pending`
- Progress is tracked per directory to support multiple batch operations
- Timestamps and detailed results are stored for each processed file

### Resume Logic
- **Default behavior**: Automatically resumes from previous progress
- **Resume mode**: Explicitly resume, skipping already completed files
- **Force restart**: Ignore previous progress and process all files
- **Clear progress**: Remove tracking data for a directory

## New Command Line Options

### Batch Upload Options

#### `--resume`
- **Description**: Explicitly resume from previous batch upload
- **Behavior**: Skip already processed files, continue with pending ones
- **Use case**: When you want to be explicit about resuming

#### `--force-restart`
- **Description**: Force restart and reprocess all files
- **Behavior**: Ignore previous progress, process all files from scratch
- **Use case**: When you want to reprocess all files regardless of previous status

#### `--clear-progress`
- **Description**: Clear progress tracking for this directory before starting
- **Behavior**: Remove all tracking data for the directory, then start fresh
- **Use case**: When you want to start completely fresh

### Progress Status Command

#### `progress <directory>`
- **Description**: View batch upload progress status for a directory
- **Shows**: Completed, failed, and pending files with statistics

#### `progress <directory> --clear`
- **Description**: Clear progress for a specific directory
- **Use case**: Reset progress tracking for a directory

## Usage Examples

### Basic Resume (Default Behavior)
```bash
# First run - processes all files
python -m src.core.cli batch-upload /papers/

# Interrupted by error/system shutdown...

# Second run - automatically resumes, skips completed files
python -m src.core.cli batch-upload /papers/
```

### Explicit Resume Mode
```bash
# Explicitly resume from previous progress
python -m src.core.cli batch-upload /papers/ --resume
```

### Force Restart (Ignore Previous Progress)
```bash
# Force restart and process all files again
python -m src.core.cli batch-upload /papers/ --force-restart
```

### Clear Progress and Start Fresh
```bash
# Clear previous progress and start fresh
python -m src.core.cli batch-upload /papers/ --clear-progress
```

### Check Progress Status
```bash
# View current progress for a directory
python -m src.core.cli progress /papers/

# Clear progress for a directory
python -m src.core.cli progress /papers/ --clear
```

### Combined Options
```bash
# Resume with custom processors
python -m src.core.cli batch-upload /papers/ --resume --processors 8

# Force restart with sequential processing
python -m src.core.cli batch-upload /papers/ --force-restart --sequential

# Clear progress and use multiprocessing
python -m src.core.cli batch-upload /papers/ --clear-progress --processors 6
```

## Progress Tracking File

### Location
- **File**: `data/batch_upload_tracker.json`
- **Format**: JSON with file paths as keys
- **Structure**: Preserves progress for multiple directories

### Example Tracking Data
```json
{
  "/papers/paper1.pdf": {
    "status": "completed",
    "directory": "/papers/",
    "paper_id": "abc123",
    "total_sentences": 150,
    "total_citations": 45,
    "completed_at": "2024-01-15 10:30:00",
    "error": null
  },
  "/papers/paper2.pdf": {
    "status": "failed",
    "directory": "/papers/",
    "paper_id": null,
    "total_sentences": 0,
    "total_citations": 0,
    "completed_at": "2024-01-15 10:35:00",
    "error": "PDF processing failed: Invalid PDF format"
  }
}
```

## Progress Summary Display

When resuming, the system shows a progress summary:

```
📊 Progress Summary:
   Previously completed: 15
   Previously failed: 2
   Success rate: 88.2%
   Files to process: 8
```

## Error Handling

### Failed Files
- Failed files are tracked and can be retried
- Error messages are stored for debugging
- Failed files are included in progress statistics

### Interruption Recovery
- Progress is saved after each file completion
- System can resume from any interruption point
- No data loss during unexpected shutdowns

## Best Practices

### For Large Batches
- Use `--resume` for explicit control over resuming
- Monitor progress with `progress` command
- Use `--clear-progress` to start fresh when needed

### For Debugging
- Check failed files in progress tracking
- Use `--sequential` with `--force-restart` for detailed error messages
- Clear progress when starting with fixed issues

### For Production
- Default resume behavior handles most interruptions
- Use `--force-restart` sparingly (reprocesses all files)
- Monitor progress files for disk space

## Troubleshooting

### Progress File Issues
- **Corrupted file**: Delete `data/batch_upload_tracker.json` to reset all progress
- **Permission errors**: Check write permissions for `data/` directory
- **Disk space**: Monitor file size, clear old progress if needed

### Resume Issues
- **Stuck on failed files**: Use `--force-restart` to retry all files
- **Wrong progress**: Use `--clear-progress` to reset for specific directory
- **Missing files**: Progress is tied to file paths, moving files may affect tracking

### Performance Considerations
- Progress tracking adds minimal overhead
- JSON file grows with number of processed files
- Consider clearing old progress for very large datasets 