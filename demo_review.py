import os
import sqlite3

DB_PASSWORD = "admin123"

def run_backup(folder):
    os.system("tar -czf backup.tar.gz " + folder)

def find_user(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '" + username + "'")
    return cur.fetchall()
