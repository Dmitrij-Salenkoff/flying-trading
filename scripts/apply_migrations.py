#!/usr/bin/env python3
"""
Script to apply ClickHouse migrations.
Usage: python scripts/apply_migrations.py
"""

import sys
from pathlib import Path

from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError

from flying_trading.config import config

# Add project root to path
project_root = Path(__file__).parent.parent


def apply_migrations():
    """Apply all ClickHouse migrations in order."""
    migrations_dir = project_root / "migrations" / "clickhouse"

    if not migrations_dir.exists():
        print(f"Migrations directory not found: {migrations_dir}")
        return False

    # Get all migration files sorted by name
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("No migration files found")
        return False

    # Connect to ClickHouse
    try:
        client = Client.from_url(config.clickhouse_url)
        print(f"Connected to ClickHouse: {config.clickhouse_url}")
    except Exception as e:
        print(f"Failed to connect to ClickHouse: {e}")
        return False

    # Apply each migration
    for migration_file in migration_files:
        print(f"\nApplying migration: {migration_file.name}")
        try:
            with open(migration_file, encoding="utf-8") as f:
                sql_content = f.read()

            # Split SQL by semicolon and filter out empty/comment-only statements
            statements = [
                stmt.strip()
                for stmt in sql_content.split(";")
                if stmt.strip() and not stmt.strip().startswith("--")
            ]

            # Execute each statement separately
            for statement in statements:
                if statement:  # Skip empty statements
                    client.execute(statement)

            print(f"✓ Successfully applied {migration_file.name}")
        except ClickHouseError as e:
            # Check if error is about table/database already existing
            error_msg = str(e).lower()
            if "already exists" in error_msg or "exists" in error_msg:
                print(f"⚠ {migration_file.name} already applied (skipping)")
            else:
                print(f"✗ Error applying {migration_file.name}: {e}")
                return False
        except Exception as e:
            print(f"✗ Error applying {migration_file.name}: {e}")
            return False

    print("\n✓ All migrations applied successfully!")
    return True


if __name__ == "__main__":
    success = apply_migrations()
    sys.exit(0 if success else 1)
