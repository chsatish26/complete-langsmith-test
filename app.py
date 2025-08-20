# Step 1: Create a simple app.py file (this is the standard LangGraph Cloud pattern)

# app.py
"""
Simple LangGraph Cloud deployment for LangSmith evaluation
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, TypedDict
import uuid

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

# ==================== STATE DEFINITION ====================

class EvaluationState(TypedDict):
    """State for the evaluation workflow"""
    input_data: Dict[str, Any]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    step: str

# ==================== SIMPLE TEST DATA ====================

QUICK_TESTS = {
    "conversation_test": {
        "messages": [
            "I need help with my account",
            "The login isn't working",
            "This is urgent for my business"
        ],
        "expected_keywords": ["account", "login", "urgent"]
    },
    "personalization_test": {
        "user_profile": {
            "name": "Alex",
            "role": "Developer", 
            "level": "expert"
        },
        "query": "How do I optimize API performance?"
    }
}

# ==================== SIMPLE NODES ====================

class SimpleEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    @traceable(name="conversation_test")
    async def test_conversation(self, state: EvaluationState) -> EvaluationState:
        """Test basic conversation flow"""
        print("🧠 Testing conversation...")
        
        test_data = QUICK_TESTS["conversation_test"]
        conversation_context = ""
        results = []
        
        for i, message in enumerate(test_data["messages"]):
            start_time = time.time()
            
            if conversation_context:
                prompt = f"Previous: {conversation_context}\nUser: {message}\nRespond naturally:"
            else:
                prompt = f"User: {message}\nRespond as a helpful assistant:"
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            conversation_context += f" {message}"
            
            end_time = time.time()
            
            # Check if context is maintained
            context_maintained = any(
                keyword.lower() in response.content.lower() 
                for keyword in test_data["expected_keywords"]
            ) if i > 0 else True
            
            results.append({
                "message_num": i + 1,
                "message": message,
                "response_length": len(response.content),
                "context_maintained": context_maintained,
                "latency_ms": (end_time - start_time) * 1000,
                "success": context_maintained
            })
        
        state["results"].extend(results)
        state["step"] = "personalization"
        return state

    @traceable(name="personalization_test")
    async def test_personalization(self, state: EvaluationState) -> EvaluationState:
        """Test user personalization"""
        print("👤 Testing personalization...")
        
        test_data = QUICK_TESTS["personalization_test"]
        profile = test_data["user_profile"]
        
        start_time = time.time()
        
        prompt = f"""
        User Profile: {profile["name"]} is a {profile["role"]} with {profile["level"]} level expertise.
        
        Question: {test_data["query"]}
        
        Respond in a way appropriate for their expertise level.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        end_time = time.time()
        
        # Simple personalization check
        personalization_score = 0
        if profile["name"].lower() in response.content.lower():
            personalization_score += 1
        if profile["level"] == "expert" and ("technical" in response.content.lower() or "advanced" in response.content.lower()):
            personalization_score += 1
        if len(response.content) > 200:  # Detailed response for expert
            personalization_score += 1
        
        result = {
            "test_type": "personalization",
            "user_profile": profile,
            "query": test_data["query"],
            "response_length": len(response.content),
            "personalization_score": personalization_score,
            "max_score": 3,
            "success": personalization_score >= 2,
            "latency_ms": (end_time - start_time) * 1000
        }
        
        state["results"].append(result)
        state["step"] = "summary"
        return state

    @traceable(name="generate_summary")
    async def generate_summary(self, state: EvaluationState) -> EvaluationState:
        """Generate test summary"""
        print("📊 Generating summary...")
        
        results = state["results"]
        
        # Calculate metrics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get("success", True))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        avg_latency = sum(r.get("latency_ms", 0) for r in results) / total_tests if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "timestamp": datetime.now().isoformat(),
            "status": "excellent" if success_rate >= 0.9 else "good" if success_rate >= 0.7 else "needs_improvement",
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "langgraph-simple-evaluation"),
            "dashboard_url": f"https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'langgraph-simple-evaluation')}"
        }
        
        state["summary"] = summary
        state["step"] = "complete"
        return state

# ==================== GRAPH CREATION ====================

def create_evaluation_graph():
    """Create simple evaluation graph"""
    
    evaluator = SimpleEvaluator()
    
    # Create workflow
    workflow = StateGraph(EvaluationState)
    
    # Add nodes
    workflow.add_node("conversation", evaluator.test_conversation)
    workflow.add_node("personalization", evaluator.test_personalization)
    workflow.add_node("summary", evaluator.generate_summary)
    
    # Add edges
    workflow.set_entry_point("conversation")
    workflow.add_edge("conversation", "personalization")
    workflow.add_edge("personalization", "summary")
    workflow.add_edge("summary", END)
    
    # Compile
    checkpointer = SqliteSaver.from_conn_string(":memory:")
    return workflow.compile(checkpointer=checkpointer)

# ==================== MAIN ENTRY POINT ====================

@traceable(name="langsmith_evaluation")
async def run_evaluation(input_data: Dict[str, Any] = None):
    """Main evaluation function"""
    
    # Initialize state
    initial_state: EvaluationState = {
        "input_data": input_data or {},
        "results": [],
        "summary": {},
        "step": "conversation"
    }
    
    # Create and run graph
    app = create_evaluation_graph()
    
    try:
        final_state = await app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        
        return {
            "success": True,
            "summary": final_state["summary"],
            "results": final_state["results"],
            "message": "LangSmith evaluation completed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Evaluation failed"
        }

# ==================== FOR LOCAL TESTING ====================

async def main():
    """For local testing"""
    print("🚀 Simple LangSmith Evaluation")
    print("=" * 40)
    
    result = await run_evaluation()
    
    if result["success"]:
        summary = result["summary"]
        print(f"✅ Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️ Avg Latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"📊 Status: {summary['status']}")
        print(f"🔗 Dashboard: {summary['dashboard_url']}")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

---

# requirements.txt
langchain 
langgraph
langsmith
langchain-openai

---

# langgraph.json - SIMPLE VERSION THAT WORKS
{
  "dependencies": [
    "langchain",
    "langgraph",
    "langsmith",
    "langchain-openai"
  ],
  "graphs": {
    "evaluation": "./app.py:create_evaluation_graph"
  },
  "env": [
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT"
  ]
}
