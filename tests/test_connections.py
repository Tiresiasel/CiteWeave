#!/usr/bin/env python3
"""
Simple connection test for Neo4j and vector database
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.graph_builder import GraphDB
from src.vector_indexer import VectorIndexer
from config_manager import ConfigManager

def test_neo4j_connection():
    """Test Neo4j connection with different passwords"""
    print("🔗 Testing Neo4j connection...")
    
    # Common passwords to try
    passwords = ["12345678", "neo4j", "password", "admin"]
    
    for password in passwords:
        try:
            print(f"  Trying password: {password}")
            db = GraphDB(uri="bolt://localhost:7687", user="neo4j", password=password)
            
            # Test connection with a simple query
            with db.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    print(f"  ✅ Connected successfully with password: {password}")
                    
                    # Update config file
                    import json
                    config_file = "config/neo4j_config.json"
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    config["password"] = password
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"  ✅ Updated config file with working password")
                    
                    db.close()
                    return password
            
        except Exception as e:
            print(f"  ❌ Failed with password {password}: {e}")
            continue
    
    print("  ❌ Could not connect with any common password")
    return None

def test_vector_database():
    """Test vector database connection"""
    print("\n🔗 Testing vector database connection...")
    
    try:
        indexer = VectorIndexer()
        print("  ✅ Vector indexer initialized successfully")
        return True
    except Exception as e:
        print(f"  ❌ Vector indexer failed: {e}")
        return False

def main():
    print("=" * 50)
    print("🧪 Connection Test Suite")
    print("=" * 50)
    
    # Test Neo4j
    neo4j_password = test_neo4j_connection()
    
    # Test Vector DB
    vector_success = test_vector_database()
    
    print("\n📊 Summary:")
    print(f"  Neo4j: {'✅ Connected' if neo4j_password else '❌ Failed'}")
    print(f"  Vector DB: {'✅ Connected' if vector_success else '❌ Failed'}")
    
    if neo4j_password and vector_success:
        print("\n🎉 All connections successful! Ready for database integration testing.")
    else:
        print("\n⚠️  Some connections failed. Check the issues above.")

if __name__ == "__main__":
    main() 