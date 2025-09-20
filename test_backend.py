#!/usr/bin/env python3
"""
Test script to debug backend issues
"""
import requests
import time
import os

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://127.0.0.1:8001/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_small_file():
    """Test with a small text file"""
    print("\n🧪 Testing with small text file...")
    
    # Create a small test file
    test_content = "This is a test legal document. It contains some sample text for testing purposes."
    
    files = {"file": ("test.txt", test_content.encode(), "text/plain")}
    
    try:
        print("📤 Uploading test file...")
        start_time = time.time()
        
        response = requests.post("http://127.0.0.1:8000/parse", files=files, timeout=60)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Test file processed successfully in {duration:.2f} seconds")
            print(f"📊 Result: {result}")
            return True
        else:
            print(f"❌ Test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Test timed out - this indicates a performance issue")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    print("🔍 Backend Debug Test")
    print("=" * 50)
    
    # Test 1: Backend health
    if not test_backend_health():
        print("\n💡 Make sure to start the backend with:")
        print("   cd backend")
        print("   uvicorn app:app --reload")
        return
    
    # Test 2: Small file upload
    if test_small_file():
        print("\n✅ Backend is working correctly with small files")
        print("💡 The issue might be with your specific PDF file or its size")
    else:
        print("\n❌ Backend has issues even with small files")
        print("💡 Check the backend logs for detailed error messages")

if __name__ == "__main__":
    main()
