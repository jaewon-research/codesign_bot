#!/usr/bin/env python3
"""
Simple web-based SQLite database viewer for the Reddit simulation results.
"""

import sqlite3
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import threading
import time

class DatabaseViewer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Get database summary
            conn = sqlite3.connect('reddit_bedrock_simulation.db')
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get counts
            counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                counts[table] = cursor.fetchone()[0]
            
            # Get sample data
            sample_data = {}
            for table in tables:
                cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                sample_data[table] = {
                    'columns': columns,
                    'rows': rows
                }
            
            conn.close()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Reddit Bedrock Simulation Database</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .table {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                    .table-header {{ background: #f5f5f5; padding: 10px; font-weight: bold; }}
                    .table-content {{ padding: 10px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>🔍 Reddit Bedrock Simulation Database</h1>
                
                <div class="summary">
                    <h2>📊 Summary</h2>
                    <ul>
                        {''.join([f'<li><strong>{table}:</strong> {count} records</li>' for table, count in counts.items()])}
                    </ul>
                </div>
                
                {''.join([f'''
                <div class="table">
                    <div class="table-header">
                        <h2>📋 {table.upper()} ({counts[table]} records)</h2>
                    </div>
                    <div class="table-content">
                        <table>
                            <thead>
                                <tr>
                                    {''.join([f'<th>{col}</th>' for col in sample_data[table]['columns']])}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([f'<tr>{"".join([f"<td>{str(cell)}</td>" for cell in row])}</tr>' for row in sample_data[table]['rows']])}
                            </tbody>
                        </table>
                    </div>
                </div>
                ''' for table in tables])}
                
                <div style="margin-top: 30px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
                    <p><strong>💡 Tip:</strong> This is a sample view. For full database access, use SQLite command line or VS Code extensions.</p>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_server():
    """Start the web server."""
    server = HTTPServer(('localhost', 8080), DatabaseViewer)
    print("🌐 Database viewer started at http://localhost:8080")
    print("📊 Opening database in browser...")
    webbrowser.open('http://localhost:8080')
    server.serve_forever()

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n👋 Database viewer stopped.")
