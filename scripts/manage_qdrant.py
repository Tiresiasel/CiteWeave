#!/usr/bin/env python3
"""
Management script for Qdrant server operations
"""

import argparse
import subprocess
import sys
import time
import requests
from pathlib import Path

def check_server_status():
    """Check if Qdrant server is running"""
    try:
        response = requests.get("http://localhost:6333/", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant server is running")
            return True
        else:
            print("❌ Qdrant server is not responding properly")
            return False
    except:
        print("❌ Qdrant server is not running")
        return False

def start_server():
    """Start Qdrant server"""
    print("🚀 Starting Qdrant server...")
    try:
        result = subprocess.run(['docker-compose', 'up', '-d', 'qdrant'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Qdrant server started")
            
            # Wait for server to be ready
            print("⏳ Waiting for server to be ready...")
            for i in range(30):
                if check_server_status():
                    return True
                time.sleep(2)
            
            print("⚠️  Server started but may not be fully ready yet")
            return True
        else:
            print(f"❌ Failed to start server: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def stop_server():
    """Stop Qdrant server"""
    print("🛑 Stopping Qdrant server...")
    try:
        result = subprocess.run(['docker-compose', 'down'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Qdrant server stopped")
            return True
        else:
            print(f"❌ Failed to stop server: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        return False

def restart_server():
    """Restart Qdrant server"""
    print("🔄 Restarting Qdrant server...")
    if stop_server():
        time.sleep(2)
        return start_server()
    return False

def show_logs():
    """Show Qdrant server logs"""
    print("📋 Showing Qdrant server logs...")
    try:
        subprocess.run(['docker-compose', 'logs', '-f', 'qdrant'])
    except KeyboardInterrupt:
        print("\n📋 Log viewing stopped")
    except Exception as e:
        print(f"❌ Error showing logs: {e}")

def show_info():
    """Show server information"""
    print("📊 Qdrant Server Information:")
    print("=" * 40)
    
    # Check if server is running
    if check_server_status():
        try:
            # Get collections info
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json()
                print(f"📚 Collections: {len(collections.get('collections', []))}")
                for collection in collections.get('collections', []):
                    name = collection.get('name', 'Unknown')
                    points = collection.get('points_count', 0)
                    print(f"   - {name}: {points} points")
            
            # Get server info
            response = requests.get("http://localhost:6333/", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"🔗 API Version: {info.get('version', 'Unknown')}")
                print(f"📊 Status: {info.get('status', 'Unknown')}")
            
        except Exception as e:
            print(f"⚠️  Could not fetch detailed info: {e}")
    else:
        print("❌ Server is not running")
    
    print("\n🔗 Access URLs:")
    print("   - REST API: http://localhost:6333")
    print("   - Web UI: http://localhost:6333/dashboard")
    print("   - Health Check: http://localhost:6333/health")

def backup_data():
    """Backup Qdrant data"""
    print("💾 Creating backup of Qdrant data...")
    try:
        backup_dir = Path("./backups/qdrant_backup")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Stop server to ensure data consistency
        print("   Stopping server for backup...")
        stop_server()
        time.sleep(2)
        
        # Copy data
        result = subprocess.run([
            'docker', 'run', '--rm', '-v', 'citeweave_qdrant_storage:/data',
            '-v', f'{backup_dir.absolute()}:/backup', 'alpine',
            'sh', '-c', 'cp -r /data/* /backup/'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Backup created at {backup_dir}")
        else:
            print(f"❌ Backup failed: {result.stderr}")
            return False
        
        # Restart server
        print("   Restarting server...")
        return start_server()
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

def restore_data():
    """Restore Qdrant data from backup"""
    backup_dir = Path("./backups/qdrant_backup")
    if not backup_dir.exists():
        print("❌ No backup found. Please create a backup first.")
        return False
    
    print("🔄 Restoring Qdrant data from backup...")
    try:
        # Stop server
        print("   Stopping server...")
        stop_server()
        time.sleep(2)
        
        # Restore data
        result = subprocess.run([
            'docker', 'run', '--rm', '-v', 'citeweave_qdrant_storage:/data',
            '-v', f'{backup_dir.absolute()}:/backup', 'alpine',
            'sh', '-c', 'rm -rf /data/* && cp -r /backup/* /data/'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data restored successfully")
        else:
            print(f"❌ Restore failed: {result.stderr}")
            return False
        
        # Restart server
        print("   Restarting server...")
        return start_server()
        
    except Exception as e:
        print(f"❌ Error restoring data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage Qdrant server for CiteWeave")
    parser.add_argument('action', choices=[
        'start', 'stop', 'restart', 'status', 'logs', 'info', 'backup', 'restore'
    ], help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        success = start_server()
        sys.exit(0 if success else 1)
    elif args.action == 'stop':
        success = stop_server()
        sys.exit(0 if success else 1)
    elif args.action == 'restart':
        success = restart_server()
        sys.exit(0 if success else 1)
    elif args.action == 'status':
        success = check_server_status()
        sys.exit(0 if success else 1)
    elif args.action == 'logs':
        show_logs()
    elif args.action == 'info':
        show_info()
    elif args.action == 'backup':
        success = backup_data()
        sys.exit(0 if success else 1)
    elif args.action == 'restore':
        success = restore_data()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 