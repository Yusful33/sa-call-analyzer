#!/usr/bin/env python3
"""
Verify Arize credentials and space access by testing the API directly.
"""
import os
import sys
import requests
from dotenv import load_dotenv
import base64

load_dotenv(override=True)

def main():
    api_key = os.getenv("ARIZE_API_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")

    print("=" * 80)
    print("ARIZE CREDENTIALS VERIFICATION")
    print("=" * 80)

    print(f"\nAPI Key: {api_key[:20]}...{api_key[-10:]}")
    print(f"Space ID: {space_id}")

    # Decode space ID
    try:
        decoded = base64.b64decode(space_id).decode('utf-8')
        print(f"Decoded: {decoded}")

        # Extract numeric ID from decoded string (format: "Space:31541:RKeK")
        parts = decoded.split(':')
        if len(parts) >= 2:
            numeric_id = parts[1]
            print(f"Numeric Space ID: {numeric_id}")
    except Exception as e:
        print(f"Failed to decode: {e}")
        sys.exit(1)

    print(f"\n" + "-" * 80)
    print("Testing Arize API Access...")
    print("-" * 80)

    # Test 1: Try to access the Arize API with these credentials
    # Note: Arize's public API endpoints may be limited, so we'll test what we can

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Try different Arize API endpoints
    endpoints_to_test = [
        ("https://api.arize.com/v1/spaces", "List Spaces"),
        (f"https://api.arize.com/v1/spaces/{numeric_id}", "Get Space Details"),
        ("https://api.arize.com/graphql", "GraphQL Endpoint"),
    ]

    for url, description in endpoints_to_test:
        print(f"\nTesting: {description}")
        print(f"URL: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

            if response.status_code == 200:
                print("✅ Success!")
            elif response.status_code == 401:
                print("❌ 401 Unauthorized - API key may be invalid")
            elif response.status_code == 403:
                print("⚠️  403 Forbidden - API key valid but no access to this resource")
            elif response.status_code == 404:
                print("⚠️  404 Not Found - Endpoint or resource doesn't exist")
            else:
                print(f"⚠️  Unexpected status code")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    print(f"\n📝 Important Notes:")
    print(f"  • Arize uses OTLP for trace ingestion (not REST API)")
    print(f"  • Even if REST API calls fail, OTLP traces may still work")
    print(f"  • The telemetry is sent to: otlp.arize.com:443")
    print(f"\n🔍 To verify traces in Arize UI:")
    print(f"  1. Go to: https://app.arize.com/")
    print(f"  2. Select your organization")
    print(f"  3. Select the space: {decoded}")
    print(f"  4. Go to 'Tracing' or 'Projects'")
    print(f"  5. Look for project: 'arize-detailed-test'")
    print(f"  6. Filter by last 24 hours")
    print(f"\n💡 Tip: Check if you're in the correct organization/space!")

if __name__ == "__main__":
    main()
