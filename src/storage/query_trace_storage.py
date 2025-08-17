import sqlite3
import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class QueryTraceStorage:
    """Query trace storage system - stores traces by query granularity"""
    
    def __init__(self, db_path=None):
        # Use environment variable or default path
        if db_path is None:
            # Use /app/data/logs for Docker compatibility (bind mount to host data directory)
            if os.path.exists("/app/data/logs"):
                db_path = "/app/data/logs/query_traces.sqlite"
            else:
                # Fallback to data/logs for local development
                data_dir = os.environ.get("CITEWEAVE_DATA_DIR", "data")
                db_path = os.path.join(data_dir, "logs", "query_traces.sqlite")
        
        self.db_path = db_path
        self._ensure_logs_directory()
        self._init_database()
        logger.info(f"QueryTraceStorage initialized with database: {self.db_path}")
    
    def _ensure_logs_directory(self):
        """Ensure logs directory exists"""
        logs_dir = os.path.dirname(self.db_path)
        if logs_dir and not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Created logs directory: {logs_dir}")
    
    def _init_database(self):
        """Initialize database table structure - OPTIMIZED 3-TABLE DESIGN"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 1. Main table: query information with enhanced metadata
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        query_id TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        query_type TEXT,
                        language TEXT,
                        workflow_config TEXT,
                        agent_config TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME,
                        total_steps INTEGER DEFAULT 0,
                        execution_status TEXT DEFAULT 'running'
                    )
                """)
                
                # 2. Execution steps table (merged workflow_trace and query_steps)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_steps (
                        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT NOT NULL,
                        step_order INTEGER NOT NULL,
                        agent_name TEXT NOT NULL,
                        workflow_step TEXT NOT NULL,
                        routing_decision TEXT,
                        input_state TEXT,
                        output_state TEXT,
                        decision_reasoning TEXT,
                        next_agent TEXT,
                        execution_time REAL,
                        memory_usage INTEGER,
                        error_message TEXT,
                        agent_state TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE
                    )
                """)
                
                # 3. Agent interactions table with step references
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_interactions (
                        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT NOT NULL,
                        from_step_id INTEGER,
                        to_step_id INTEGER,
                        interaction_type TEXT NOT NULL,
                        data_passed TEXT,
                        routing_condition TEXT,
                        interaction_metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
                        FOREIGN KEY (from_step_id) REFERENCES execution_steps (step_id) ON DELETE CASCADE,
                        FOREIGN KEY (to_step_id) REFERENCES execution_steps (step_id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for better query performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_thread_id ON queries(thread_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_user_id ON queries(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_status ON queries(execution_status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_steps_query_id ON execution_steps(query_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_steps_step_order ON execution_steps(query_id, step_order)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_interactions_query_id ON agent_interactions(query_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_interactions_steps ON agent_interactions(from_step_id, to_step_id)")
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def start_query_trace(self, query_text: str, thread_id: str, user_id: str, 
                         query_type: str = None, language: str = "en",
                         workflow_config: Dict = None, agent_config: Dict = None) -> str:
        """Start tracing a new query with enhanced configuration"""
        try:
            query_id = self._generate_query_id(query_text, thread_id, user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO queries (query_id, query_text, thread_id, user_id, query_type, language, workflow_config, agent_config)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (query_id, query_text, thread_id, user_id, query_type, language,
                      json.dumps(workflow_config, ensure_ascii=False) if workflow_config else None,
                      json.dumps(agent_config, ensure_ascii=False) if agent_config else None))
            
            logger.info(f"Started query trace: {query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to start query trace: {e}")
            raise
    
    def add_execution_step(self, query_id: str, step_order: int, agent_name: str,
                          workflow_step: str, routing_decision: str = None,
                          input_state: Dict = None, output_state: Dict = None,
                          decision_reasoning: str = None, next_agent: str = None,
                          execution_time: float = 0.0, memory_usage: int = 0,
                          error_message: str = None, agent_state: Dict = None) -> int:
        """Add execution step for a query (merged workflow_trace and query_steps)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO execution_steps 
                    (query_id, step_order, agent_name, workflow_step, routing_decision,
                     input_state, output_state, decision_reasoning, next_agent,
                     execution_time, memory_usage, error_message, agent_state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, step_order, agent_name, workflow_step, routing_decision,
                    json.dumps(input_state, ensure_ascii=False) if input_state else None,
                    json.dumps(output_state, ensure_ascii=False) if output_state else None,
                    decision_reasoning, next_agent, execution_time, memory_usage,
                    error_message, json.dumps(agent_state, ensure_ascii=False) if agent_state else None
                ))
                
                step_id = cursor.lastrowid
                
                # Update total steps count for the query
                conn.execute("""
                    UPDATE queries SET total_steps = (
                        SELECT COUNT(*) FROM execution_steps WHERE query_id = ?
                    ) WHERE query_id = ?
                """, (query_id, query_id))
                
                conn.commit()
            
            logger.debug(f"Added execution step {step_order} for query {query_id}: {agent_name} -> {workflow_step}")
            return step_id
            
        except Exception as e:
            logger.error(f"Failed to add execution step: {e}")
            raise
    
    def add_agent_interaction(self, query_id: str, from_step_id: int, to_step_id: int,
                             interaction_type: str, data_passed: Dict = None,
                             routing_condition: str = None, interaction_metadata: Dict = None):
        """Add agent-to-agent interaction record with step references"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO agent_interactions 
                    (query_id, from_step_id, to_step_id, interaction_type, data_passed, routing_condition, interaction_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, from_step_id, to_step_id, interaction_type,
                    json.dumps(data_passed, ensure_ascii=False) if data_passed else None,
                    routing_condition, json.dumps(interaction_metadata, ensure_ascii=False) if interaction_metadata else None
                ))
                
            logger.debug(f"Added agent interaction: step {from_step_id} -> step {to_step_id} ({interaction_type}) for query {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to add agent interaction: {e}")
            raise
    
    def complete_query_trace(self, query_id: str, final_response: str,
                           confidence_score: float = 0.0, total_execution_time: float = 0.0,
                           total_memory_usage: int = 0, data_sources_used: List[str] = None,
                           errors: List[str] = None, warnings: List[str] = None):
        """Complete query tracing and record final results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update query status
                conn.execute("""
                    UPDATE queries 
                    SET completed_at = CURRENT_TIMESTAMP, execution_status = 'completed'
                    WHERE query_id = ?
                """, (query_id,))
                
                # Update query status (no separate results table needed)
                conn.execute("""
                    UPDATE queries 
                    SET completed_at = CURRENT_TIMESTAMP, execution_status = 'completed'
                    WHERE query_id = ?
                """, (query_id,))
            
            logger.info(f"Completed query trace: {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to complete query trace: {e}")
            raise
    
    def get_query_trace(self, query_id: str) -> Optional[Dict]:
        """Get complete query trace information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get query basic information
                query_info = conn.execute("""
                    SELECT * FROM queries WHERE query_id = ?
                """, (query_id,)).fetchone()
                
                if not query_info:
                    logger.warning(f"Query not found: {query_id}")
                    return None
                
                # Get execution steps
                steps = conn.execute("""
                    SELECT * FROM execution_steps 
                    WHERE query_id = ? 
                    ORDER BY step_order
                """, (query_id,)).fetchall()
                
                return {
                    "query_info": dict(query_info),
                    "steps": [dict(step) for step in steps]
                }
                
        except Exception as e:
            logger.error(f"Failed to get query trace: {e}")
            return None
    
    def get_session_queries(self, thread_id: str, limit: int = 50) -> List[Dict]:
        """Get all queries in a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                queries = conn.execute("""
                    SELECT q.*
                    FROM queries q
                    WHERE q.thread_id = ?
                    ORDER BY q.created_at DESC
                    LIMIT ?
                """, (thread_id, limit)).fetchall()
                
                return [dict(query) for query in queries]
                
        except Exception as e:
            logger.error(f"Failed to get session queries: {e}")
            return []
    
    def get_agent_performance_stats(self, agent_name: str, days: int = 7) -> Dict:
        """Get performance statistics for a specific agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(execution_time) as avg_execution_time,
                        MAX(execution_time) as max_execution_time,
                        MIN(execution_time) as min_execution_time,
                        AVG(memory_usage) as avg_memory_usage,
                        COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as error_count,
                        SUM(execution_time) as total_execution_time,
                        SUM(memory_usage) as total_memory_usage
                    FROM execution_steps 
                    WHERE agent_name = ? 
                    AND created_at >= datetime('now', '-{} days')
                """.format(days), (agent_name,)).fetchone()
                
                return dict(stats)
                
        except Exception as e:
            logger.error(f"Failed to get agent performance stats: {e}")
            return {}
    
    def get_execution_steps(self, query_id: str) -> List[Dict]:
        """Get all execution steps for a query (merged workflow_trace and query_steps)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        step_id, step_order, agent_name, workflow_step, routing_decision,
                        input_state, output_state, decision_reasoning, next_agent,
                        execution_time, memory_usage, error_message, agent_state, created_at
                    FROM execution_steps 
                    WHERE query_id = ? 
                    ORDER BY step_order, created_at
                """, (query_id,))
                
                steps = []
                for row in cursor.fetchall():
                    steps.append({
                        "step_id": row[0],
                        "step_order": row[1],
                        "agent_name": row[2],
                        "workflow_step": row[3],
                        "routing_decision": row[4],
                        "input_state": json.loads(row[5]) if row[5] else None,
                        "output_state": json.loads(row[6]) if row[6] else None,
                        "decision_reasoning": row[7],
                        "next_agent": row[8],
                        "execution_time": row[9],
                        "memory_usage": row[10],
                        "error_message": row[11],
                        "agent_state": json.loads(row[12]) if row[12] else None,
                        "created_at": row[13]
                    })
                
                return steps
                
        except Exception as e:
            logger.error(f"Failed to get execution steps: {e}")
            return []
    
    def get_agent_interactions(self, query_id: str) -> List[Dict]:
        """Get agent-to-agent interactions for a query with step details"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        ai.interaction_id, ai.from_step_id, ai.to_step_id, ai.interaction_type,
                        ai.data_passed, ai.routing_condition, ai.interaction_metadata, ai.created_at,
                        es1.agent_name as from_agent, es2.agent_name as to_agent
                    FROM agent_interactions ai
                    LEFT JOIN execution_steps es1 ON ai.from_step_id = es1.step_id
                    LEFT JOIN execution_steps es2 ON ai.to_step_id = es2.step_id
                    WHERE ai.query_id = ? 
                    ORDER BY ai.created_at
                """, (query_id,))
                
                interactions = []
                for row in cursor.fetchall():
                    interactions.append({
                        "interaction_id": row[0],
                        "from_step_id": row[1],
                        "to_step_id": row[2],
                        "interaction_type": row[3],
                        "data_passed": json.loads(row[4]) if row[4] else None,
                        "routing_condition": row[5],
                        "interaction_metadata": json.loads(row[6]) if row[6] else None,
                        "created_at": row[7],
                        "from_agent": row[8],
                        "to_agent": row[9]
                    })
                
                return interactions
                
        except Exception as e:
            logger.error(f"Failed to get agent interactions: {e}")
            return []
    
    def get_complete_workflow_analysis(self, query_id: str) -> Dict:
        """Get complete workflow analysis including all traces and interactions"""
        try:
            execution_steps = self.get_execution_steps(query_id)
            agent_interactions = self.get_agent_interactions(query_id)
            
            # Analyze workflow patterns
            workflow_pattern = []
            for step in execution_steps:
                workflow_pattern.append({
                    "step": step["workflow_step"],
                    "agent": step["agent_name"],
                    "decision": step["routing_decision"],
                    "reasoning": step["decision_reasoning"],
                    "next": step["next_agent"],
                    "execution_time": step["execution_time"],
                    "memory_usage": step["memory_usage"]
                })
            
            # Analyze agent communication
            communication_flow = []
            for interaction in agent_interactions:
                communication_flow.append({
                    "from": interaction["from_agent"],
                    "to": interaction["to_agent"],
                    "type": interaction["interaction_type"],
                    "data": interaction["data_passed"],
                    "metadata": interaction["interaction_metadata"]
                })
            
            # Calculate performance metrics
            total_execution_time = sum(step.get("execution_time", 0) or 0 for step in execution_steps)
            total_memory_usage = sum(step.get("memory_usage", 0) or 0 for step in execution_steps)
            error_count = len([step for step in execution_steps if step.get("error_message")])
            
            return {
                "query_id": query_id,
                "workflow_pattern": workflow_pattern,
                "communication_flow": communication_flow,
                "execution_steps": execution_steps,
                "total_workflow_steps": len(execution_steps),
                "total_interactions": len(agent_interactions),
                "total_execution_steps": len(execution_steps),
                "performance_metrics": {
                    "total_execution_time": total_execution_time,
                    "total_memory_usage": total_memory_usage,
                    "error_count": error_count,
                    "avg_execution_time": total_execution_time / len(execution_steps) if execution_steps else 0,
                    "avg_memory_usage": total_memory_usage / len(execution_steps) if execution_steps else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get complete workflow analysis: {e}")
            return {}
    
    def cleanup_old_traces(self, days: int = 30):
        """Clean up old trace records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now().replace(day=datetime.now().day - days)
                deleted_count = conn.execute("""
                    DELETE FROM queries 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),)).rowcount
                
                logger.info(f"Cleaned up {deleted_count} old trace records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old traces: {e}")
    
    def _generate_query_id(self, query_text: str, thread_id: str, user_id: str) -> str:
        """Generate unique query ID"""
        content = f"{query_text[:100]}_{thread_id}_{user_id}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total queries
                total_queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
                
                # Completed queries
                completed_queries = conn.execute("SELECT COUNT(*) FROM queries WHERE execution_status = 'completed'").fetchone()[0]
                
                # Total steps
                total_steps = conn.execute("SELECT COUNT(*) FROM execution_steps").fetchone()[0]
                
                # Database size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    "total_queries": total_queries,
                    "completed_queries": completed_queries,
                    "total_steps": total_steps,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
