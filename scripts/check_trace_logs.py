#!/usr/bin/env python3
"""
Script to check CiteWeave trace logs and verify data persistence
"""

import os
import sys
import sqlite3
from pathlib import Path

def check_trace_logs():
    """Check if trace logs are being saved correctly"""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    db_path = logs_dir / "query_traces.db"
    
    print(f"🔍 Checking trace logs in: {project_root}")
    print(f"📁 Logs directory: {logs_dir}")
    print(f"🗄️  Database file: {db_path}")
    print()
    
    # Check if logs directory exists
    if not logs_dir.exists():
        print("❌ Logs directory does not exist!")
        print("   Run: python scripts/setup_data_directories.py")
        return False
    
    # Check if database file exists
    if not db_path.exists():
        print("❌ Database file does not exist!")
        print("   This means no queries have been processed yet")
        print("   Or there was an error in the trace storage system")
        return False
    
    # Check database file size
    db_size = db_path.stat().st_size
    print(f"✅ Database file exists")
    print(f"📊 File size: {db_size} bytes ({db_size/1024:.2f} KB)")
    
    # Try to connect to database
    try:
        with sqlite3.connect(db_path) as conn:
            # Check tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📋 Database tables: {tables}")
            
            # Check queries count
            cursor = conn.execute("SELECT COUNT(*) FROM queries")
            query_count = cursor.fetchone()[0]
            print(f"🔢 Total queries: {query_count}")
            
            # Check steps count
            cursor = conn.execute("SELECT COUNT(*) FROM query_steps")
            step_count = cursor.fetchone()[0]
            print(f"📝 Total steps: {step_count}")
            
            # Show recent queries
            if query_count > 0:
                print(f"\n📋 Recent queries:")
                cursor = conn.execute("""
                    SELECT query_id, query_text, created_at, execution_status, total_steps
                    FROM queries 
                    ORDER BY created_at DESC 
                    LIMIT 3
                """)
                
                for row in cursor.fetchall():
                    query_id, query_text, created_at, status, steps = row
                    print(f"  ID: {query_id}")
                    print(f"  Query: {query_text[:60]}...")
                    print(f"  Created: {created_at}")
                    print(f"  Status: {status}")
                    print(f"  Steps: {steps}")
                    print("  ---")
                
                # Show agent performance
                print(f"\n📈 Agent performance (last 7 days):")
                cursor = conn.execute("""
                    SELECT agent_name, COUNT(*) as executions, 
                           AVG(execution_time) as avg_time
                    FROM query_steps 
                    WHERE created_at >= datetime('now', '-7 days')
                    GROUP BY agent_name
                    ORDER BY executions DESC
                """)
                
                for row in cursor.fetchall():
                    agent, executions, avg_time = row
                    print(f"  {agent}: {executions} executions, avg {avg_time:.3f}s")
            
            else:
                print("📝 No queries found in database")
                print("   Try running a query first to generate some data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return False

def check_data_directories():
    """Check if data directories are properly set up"""
    
    project_root = Path(__file__).parent.parent
    print(f"\n📁 Checking data directories in: {project_root}")
    
    directories = [
        "data",
        "data/papers", 
        "data/vector_index",
        "data/logs",
        "logs",
        "config"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (missing)")
    
    # Check for .gitkeep files
    gitkeep_dirs = ["data/papers", "data/vector_index", "data/logs", "logs"]
    for directory in gitkeep_dirs:
        gitkeep_file = project_root / directory / ".gitkeep"
        if gitkeep_file.exists():
            print(f"  📌 {directory}/.gitkeep")
        else:
            print(f"  ⚠️  {directory}/.gitkeep (missing)")

if __name__ == "__main__":
    print("🚀 CiteWeave Trace Logs Checker")
    print("=" * 50)
    
    # Check data directories
    check_data_directories()
    
    # Check trace logs
    print("\n" + "=" * 50)
    success = check_trace_logs()
    
    if success:
        print("\n🎉 Trace logs are working correctly!")
        print("   Your query data is being persisted locally")
    else:
        print("\n❌ There are issues with trace logs")
        print("   Check the errors above and fix them")
    
    print("\n💡 Next steps:")
    print("   1. If no data exists, try running a query first")
    print("   2. Check logs/query_traces.db for your query data")
    print("   3. Use the database methods to analyze your queries")
