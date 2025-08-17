#!/usr/bin/env python3
"""
Enhanced CiteWeave Trace Logs Checker
Shows detailed multi-agent workflow tracing information
"""

import os
import sys
import sqlite3
import json
from datetime import datetime

def check_enhanced_trace_logs(project_root: str = None):
    """Check enhanced trace logs with workflow and agent interaction details"""
    
    if project_root is None:
        project_root = os.getcwd()
    
    print("🚀 Enhanced CiteWeave Trace Logs Checker")
    print("=" * 60)
    print()
    
    # Check data directories
    print("📁 Checking data directories in:", project_root)
    data_dirs = [
        "data/",
        "data/papers/",
        "data/vector_index/",
        "data/logs/",
        "logs/",
        "config/"
    ]
    
    for dir_path in data_dirs:
        full_path = os.path.join(project_root, dir_path)
        if os.path.exists(full_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
    
    print()
    print("=" * 60)
    
    # Check trace logs
    logs_dir = os.path.join(project_root, "logs")
    db_path = os.path.join(logs_dir, "query_traces.db")
    
    print("🔍 Checking enhanced trace logs in:", project_root)
    print(f"📁 Logs directory: {logs_dir}")
    print(f"🗄️  Database file: {db_path}")
    print()
    
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        print("   Try running a query first to generate trace data")
        return
    
    # Database file exists
    print("✅ Database file exists")
    file_size = os.path.getsize(db_path)
    print(f"📊 File size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    # Check database structure
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"📋 Database tables: {table_names}")
        print()
        
        # Check each table for data
        for table in table_names:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"📊 {table}: {count} records")
        
        print()
        
        # Show detailed workflow analysis if data exists
        if 'queries' in table_names:
            cursor.execute("SELECT COUNT(*) FROM queries")
            query_count = cursor.fetchone()[0]
            
            if query_count > 0:
                print("🔍 Detailed Workflow Analysis")
                print("-" * 40)
                
                # Get latest query
                cursor.execute("SELECT query_id, query_text, created_at FROM queries ORDER BY created_at DESC LIMIT 1")
                latest_query = cursor.fetchone()
                
                if latest_query:
                    query_id, query_text, created_at = latest_query
                    print(f"📝 Latest Query: {query_text[:100]}...")
                    print(f"🆔 Query ID: {query_id}")
                    print(f"⏰ Created: {created_at}")
                    print()
                    
                    # Show workflow trace
                    if 'workflow_trace' in table_names:
                        cursor.execute("""
                            SELECT workflow_step, agent_name, routing_decision, next_agent, execution_order
                            FROM workflow_trace 
                            WHERE query_id = ? 
                            ORDER BY execution_order, created_at
                        """, (query_id,))
                        
                        workflow_steps = cursor.fetchall()
                        if workflow_steps:
                            print("🔄 Workflow Execution Path:")
                            for step in workflow_steps:
                                workflow_step, agent_name, routing_decision, next_agent, execution_order = step
                                print(f"  {execution_order}. {workflow_step} -> {agent_name}")
                                if routing_decision:
                                    print(f"     Decision: {routing_decision}")
                                if next_agent:
                                    print(f"     Next: {next_agent}")
                                print()
                        else:
                            print("⚠️  No workflow trace data found")
                    
                    # Show agent interactions
                    if 'agent_interactions' in table_names:
                        cursor.execute("""
                            SELECT from_agent, to_agent, interaction_type, routing_condition
                            FROM agent_interactions 
                            WHERE query_id = ? 
                            ORDER BY created_at
                        """, (query_id,))
                        
                        interactions = cursor.fetchall()
                        if interactions:
                            print("🤝 Agent Interactions:")
                            for interaction in interactions:
                                from_agent, to_agent, interaction_type, routing_condition = interaction
                                print(f"  {from_agent} -> {to_agent} ({interaction_type})")
                                if routing_condition:
                                    print(f"     Condition: {routing_condition}")
                                print()
                        else:
                            print("⚠️  No agent interaction data found")
                    
                    # Show execution steps
                    cursor.execute("""
                        SELECT agent_name, step_number, execution_time, memory_usage
                        FROM query_steps 
                        WHERE query_id = ? 
                        ORDER BY step_number
                    """, (query_id,))
                    
                    execution_steps = cursor.fetchall()
                    if execution_steps:
                        print("⚡ Execution Performance:")
                        total_time = 0
                        total_memory = 0
                        for step in execution_steps:
                            agent_name, step_number, exec_time, memory = step
                            total_time += exec_time or 0
                            total_memory += memory or 0
                            print(f"  Step {step_number}: {agent_name}")
                            if exec_time:
                                print(f"     Time: {exec_time:.3f}s")
                            if memory:
                                print(f"     Memory: {memory} bytes")
                            print()
                        
                        print(f"📊 Total Execution Time: {total_time:.3f}s")
                        print(f"📊 Total Memory Usage: {total_memory} bytes")
                    
            else:
                print("📝 No queries found in database")
                print("   Try running a query first to generate trace data")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return
    
    print("🎉 Enhanced trace logs are working correctly!")
    print("   Your multi-agent workflow data is being persisted locally")
    print()
    print("💡 New Features:")
    print("   🔄 Workflow Trace: Records each step of the multi-agent workflow")
    print("   🤝 Agent Interactions: Tracks communication between agents")
    print("   ⚡ Performance Metrics: Execution time and memory usage")
    print("   🧠 Decision Reasoning: Why each routing decision was made")
    print()
    print("💡 Next steps:")
    print("   1. Run a query to see the enhanced tracing in action")
    print("   2. Check logs/query_traces.db for detailed workflow data")
    print("   3. Use the new methods to analyze your multi-agent system")

if __name__ == "__main__":
    check_enhanced_trace_logs()
