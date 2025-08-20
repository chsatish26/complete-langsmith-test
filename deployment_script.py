import os
import json
import subprocess
import sys

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if langchain CLI is installed
    try:
        result = subprocess.run(["langchain", "--version"], capture_output=True, text=True)
        print(f"✅ LangChain CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ LangChain CLI not found")
        print("   Install with: pip install langchain-cli")
        return False
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Set")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def create_deployment_files():
    """Create necessary deployment files"""
    print("📁 Creating deployment files...")
    
    # Create langgraph.json
    config = {
        "dependencies": [
            "langchain>=0.1.0",
            "langgraph>=0.0.40", 
            "langsmith>=0.1.0",
            "langchain-openai>=0.1.0"
        ],
        "graphs": {
            "langsmith_evaluation": "./langsmith_evaluation_graph.py:create_evaluation_graph"
        },
        "env": [
            "OPENAI_API_KEY",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2", 
            "LANGCHAIN_PROJECT"
        ]
    }
    
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✅ Created langgraph.json")
    
    # Create requirements.txt
    requirements = [
        "langchain>=0.1.0",
        "langgraph>=0.0.40",
        "langsmith>=0.1.0", 
        "langchain-openai>=0.1.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    print("✅ Created requirements.txt")

def deploy_to_cloud():
    """Deploy to LangGraph Cloud"""
    print("🚀 Deploying to LangGraph Cloud...")
    
    try:
        # Deploy using LangChain CLI
        result = subprocess.run([
            "langchain", "app", "deploy", 
            "--name", "langsmith-evaluation",
            "--description", "LangSmith SaaS Platform Evaluation"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Deployment successful!")
        print(result.stdout)
        
        # Extract deployment URL from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'https://' in line and 'langchain' in line:
                print(f"🌐 Deployment URL: {line.strip()}")
                break
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_deployment(deployment_url):
    """Test the deployed application"""
    print("🧪 Testing deployment...")
    
    import requests
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{deployment_url}/health")
        if health_response.status_code == 200:
            print("✅ Health check passed")
        
        # Test evaluation endpoint
        test_payload = {
            "test_type": "quick",
            "test_data": {"scenarios": 1}
        }
        
        eval_response = requests.post(
            f"{deployment_url}/langsmith_evaluation/invoke",
            json=test_payload
        )
        
        if eval_response.status_code == 200:
            print("✅ Evaluation endpoint working")
            result = eval_response.json()
            print(f"📊 Test success rate: {result.get('summary', {}).get('overall_success_rate', 'N/A')}")
        else:
            print(f"⚠️ Evaluation test failed: {eval_response.status_code}")
        
    except Exception as e:
        print(f"⚠️ Testing failed: {e}")

def main():
    """Main deployment function"""
    print("🚀 LangGraph Cloud Deployment Script")
    print("=" * 50)
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False
    
    create_deployment_files()
    
    if deploy_to_cloud():
        print("\n✅ Deployment completed successfully!")
        print("\n📋 Next steps:")
        print("1. Check LangGraph Cloud dashboard for deployment status")
        print("2. Test the evaluation endpoint")
        print("3. Monitor traces in LangSmith dashboard")
        print("4. Scale up testing with production data")
        return True
    else:
        print("\n❌ Deployment failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)