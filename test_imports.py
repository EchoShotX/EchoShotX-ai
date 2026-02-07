#!/usr/bin/env python3
"""Test script to validate all imports work correctly"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all the import chains"""
    try:
        print("Testing basic module imports...")
        
        # Test 1: Import domain models
        print("1. Testing domain.job import...")
        from echoshot_ai_server.domain.job import Job, TaskResult, JobStatus, TaskType
        print("   ✓ domain.job imported successfully")
        
        # Test 2: Import configuration
        print("2. Testing config.settings import...")
        from echoshot_ai_server.config.settings import get_settings
        print("   ✓ config.settings imported successfully")
        
        # Test 3: Import core modules individually
        print("3. Testing individual core module imports...")
        from echoshot_ai_server.core.s3_client import S3Client
        print("   ✓ S3Client imported successfully")
        
        from echoshot_ai_server.core.sqs_client import SQSClient
        print("   ✓ SQSClient imported successfully")
        
        from echoshot_ai_server.core.api_client import SpringAPIClient
        print("   ✓ SpringAPIClient imported successfully")
        
        # Test 4: Test the problematic import from __init__.py
        print("4. Testing core.__init__.py imports...")
        from echoshot_ai_server.core import ApiClient
        print("   ✓ ApiClient imported successfully via __init__.py")
        
        # Test 5: Test the full main.py import chain
        print("5. Testing main.py import chain...")
        from echoshot_ai_server.core.sqs_client import SQSClient as MainSQSClient
        from echoshot_ai_server.core.s3_client import S3Client as MainS3Client
        from echoshot_ai_server.core.api_client import SpringAPIClient as MainSpringAPIClient
        print("   ✓ main.py import chain works")
        
        # Test 6: Test that ApiClient is actually SpringAPIClient
        print("6. Testing ApiClient alias...")
        from echoshot_ai_server.core.api_client import SpringAPIClient
        from echoshot_ai_server.core import ApiClient
        
        assert ApiClient is SpringAPIClient, "ApiClient should be aliased to SpringAPIClient"
        print("   ✓ ApiClient is correctly aliased to SpringAPIClient")
        
        print("\n🎉 All imports are working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)