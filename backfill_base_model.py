"""
Backfill base_model for lora_suggestions

One-time script to fetch base_model from CivitAI API for existing lora_suggestions
that have NULL or 'Unknown' base_model values.

Usage:
    python backfill_base_model.py
"""

import os
import sys
import asyncio
from typing import Optional, Dict, Any

import aiohttp
import psycopg_pool
from dotenv import load_dotenv

# Windows event loop policy fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Configuration from environment
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", os.environ.get("API_KEY"))

# Database configuration
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

# Rate limiting - CivitAI has limits
DELAY_BETWEEN_REQUESTS = 0.5  # seconds


async def fetch_version_info(session: aiohttp.ClientSession, version_id: int) -> Optional[Dict[str, Any]]:
    """Fetch LoRA version details from CivitAI API."""
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 404:
                print(f"  Version {version_id} not found on CivitAI")
                return None
            else:
                print(f"  Failed to fetch version {version_id}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"  Error fetching version {version_id}: {e}")
        return None


async def backfill():
    """Main backfill function."""
    print("=" * 60)
    print("Backfill base_model for lora_suggestions")
    print("=" * 60)
    
    # Create database pool
    pool = psycopg_pool.AsyncConnectionPool(DSN, min_size=1, max_size=5)
    await pool.open()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get all suggestions with NULL or Unknown base_model
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT version_id, name 
                        FROM lora_suggestions 
                        WHERE (base_model IS NULL OR base_model = '' OR base_model = 'Unknown')
                          AND version_id IS NOT NULL
                        ORDER BY version_id
                    """)
                    rows = await cur.fetchall()
            
            total = len(rows)
            print(f"\nFound {total} suggestions needing base_model\n")
            
            if total == 0:
                print("Nothing to backfill!")
                return
            
            updated = 0
            failed = 0
            
            for idx, (version_id, name) in enumerate(rows, 1):
                print(f"[{idx}/{total}] Processing: {name} (version_id: {version_id})")
                
                # Fetch from CivitAI
                version_data = await fetch_version_info(session, version_id)
                
                if version_data and version_data.get("baseModel"):
                    base_model = version_data["baseModel"]
                    
                    # Update database
                    async with pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute("""
                                UPDATE lora_suggestions 
                                SET base_model = %s 
                                WHERE version_id = %s
                            """, (base_model, version_id))
                        await conn.commit()
                    
                    print(f"  ✓ Updated: {base_model}")
                    updated += 1
                else:
                    print(f"  ✗ Could not determine base_model")
                    failed += 1
                
                # Rate limiting
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            
            print("\n" + "=" * 60)
            print(f"COMPLETE: {updated} updated, {failed} failed")
            print("=" * 60)
    
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(backfill())
