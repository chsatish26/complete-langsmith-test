# langsmith_evaluation_graph.py
# LangGraph Cloud deployment for LangSmith evaluation

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, asdict
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

# ==================== GRAPH STATE DEFINITION ====================

class EvaluationState(TypedDict):
    """State for the LangSmith evaluation workflow"""
    test_type: str
    test_data: Dict[str, Any]
    current_step: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]

# ==================== TEST DATA ====================

TEST_SCENARIOS = {
    "conversation_memory": [
        {
            "name": "Customer Support Escalation",
            "messages": [
                "I can't access my dashboard since yesterday",
                "I've tried clearing cache and different browsers",
                "This is blocking my team's work - urgent help needed",
                "Can you escalate this to technical support immediately?"
            ],
            "expected_context": ["dashboard", "cache", "urgent", "technical support"]
        },
        {
            "name": "Sales Demo Request",
            "messages": [
                "We're evaluating AI platforms for our 500-person company",
                "Need to see enterprise security features and compliance",
                "Can we schedule a technical demo next week?",
                "Also need pricing for annual enterprise license"
            ],
            "expected_context": ["500-person", "enterprise", "demo", "pricing"]
        }
    ],
    
    "user_profiles": [
        {
            "user_id": "enterprise_cto",
            "name": "Alex Thompson",
            "role": "Chief Technology Officer",
            "company": "TechCorp Enterprise",
            "technical_level": "expert",
            "preferences": {
                "communication_style": "technical_detailed",
                "priority_topics": ["architecture", "security", "scalability"],
                "response_format": "structured_with_examples"
            }
        },
        {
            "user_id": "marketing_manager", 
            "name": "Sarah Davis",
            "role": "Marketing Manager",
            "company": "Growth Startup",
            "technical_level": "beginner",
            "preferences": {
                "communication_style": "friendly_simple",
                "priority_topics": ["analytics", "user_engagement", "roi"],
                "response_format": "concise_with_visuals"
            }
        }
    ],
    
    "multi_agent_tasks": [
        {
            "task_id": "competitive_analysis",
            "description": "Analyze competitive landscape for enterprise AI platforms",
            "required_agents": ["researcher", "analyst", "strategist"],
            "success_criteria": ["market_size", "key_players", "differentiation", "recommendations"]
        },
        {
            "task_id": "product_roadmap",
            "description": "Prioritize product features based on customer feedback analysis",
            "required_agents": ["data_analyst", "product_manager", "coordinator"],
            "success_criteria": ["feedback_themes", "priority_matrix", "timeline", "resource_requirements"]
        }
    ]
}

# ==================== LANGGRAPH NODES ====================

class LangSmithEvaluationGraph:
    """LangGraph Cloud compatible evaluation workflow"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
    @traceable(name="conversation_memory_evaluation")
    async def evaluate_conversation_memory(self, state: EvaluationState) -> EvaluationState:
        """Evaluate conversation memory and context retention"""
        print("🧠 Evaluating Conversation Memory...")
        
        results = []
        scenarios = TEST_SCENARIOS["conversation_memory"]
        
        for scenario in scenarios:
            conversation_context = ""
            scenario_results = []
            
            for i, message in enumerate(scenario["messages"]):
                start_time = time.time()
                
                # Build context-aware prompt
                if conversation_context:
                    prompt = f"""
                    Previous conversation: {conversation_context}
                    
                    Continue this conversation naturally. User says: "{message}"
                    Maintain context from previous messages.
                    """
                else:
                    prompt = f"User says: '{message}'. Respond helpfully as a customer service agent."
                
                # Get AI response
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                
                # Update conversation context
                conversation_context += f" User: {message[:50]}... AI: {response.content[:50]}..."
                
                end_time = time.time()
                
                # Evaluate context retention
                context_maintained = True
                if i > 0:  # Check context retention after first message
                    context_maintained = any(
                        keyword.lower() in response.content.lower() 
                        for keyword in scenario["expected_context"]
                        if len(keyword) > 3
                    )
                
                scenario_results.append({
                    "turn": i + 1,
                    "message": message,
                    "response_length": len(response.content),
                    "context_maintained": context_maintained,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": context_maintained
                })
            
            results.append({
                "scenario": scenario["name"],
                "turns": len(scenario["messages"]),
                "success_rate": sum(1 for r in scenario_results if r["success"]) / len(scenario_results),
                "avg_latency": sum(r["latency_ms"] for r in scenario_results) / len(scenario_results),
                "details": scenario_results
            })
        
        # Update state
        state["results"].extend(results)
        state["current_step"] = "user_profiles"
        
        return state
    
    @traceable(name="user_profile_evaluation")
    async def evaluate_user_profiles(self, state: EvaluationState) -> EvaluationState:
        """Evaluate user profile management and personalization"""
        print("👤 Evaluating User Profiles...")
        
        results = []
        profiles = TEST_SCENARIOS["user_profiles"]
        
        for profile in profiles:
            start_time = time.time()
            
            # Create personalized interaction
            user_query = f"I need help optimizing our {profile['preferences']['priority_topics'][0]} strategy"
            
            personalization_prompt = f"""
            User Profile:
            - Name: {profile['name']}
            - Role: {profile['role']} 
            - Company: {profile['company']}
            - Technical Level: {profile['technical_level']}
            - Communication Style: {profile['preferences']['communication_style']}
            - Priority Topics: {', '.join(profile['preferences']['priority_topics'])}
            
            Respond to: "{user_query}"
            Adapt your response to their technical level and communication preferences.
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=personalization_prompt)])
            end_time = time.time()
            
            # Evaluate personalization
            personalization_checks = {
                "name_used": profile['name'].split()[0].lower() in response.content.lower(),
                "role_relevant": any(topic.lower() in response.content.lower() 
                                   for topic in profile['preferences']['priority_topics']),
                "technical_level_appropriate": (
                    ("detailed" in response.content.lower() or "technical" in response.content.lower())
                    if profile['technical_level'] == 'expert' else
                    ("simple" in response.content.lower() or "basic" in response.content.lower())
                ),
                "communication_style_matched": (
                    ("friendly" in response.content.lower() if "friendly" in profile['preferences']['communication_style'] else True)
                )
            }
            
            personalization_score = sum(personalization_checks.values())
            
            results.append({
                "user_id": profile['user_id'],
                "user_name": profile['name'],
                "personalization_score": personalization_score,
                "max_score": len(personalization_checks),
                "success": personalization_score >= 2,
                "latency_ms": (end_time - start_time) * 1000,
                "checks": personalization_checks,
                "response_length": len(response.content)
            })
        
        state["results"].extend(results)
        state["current_step"] = "multi_agent"
        
        return state
    
    @traceable(name="multi_agent_evaluation")
    async def evaluate_multi_agent(self, state: EvaluationState) -> EvaluationState:
        """Evaluate multi-agent coordination"""
        print("🤖 Evaluating Multi-Agent Workflows...")
        
        results = []
        tasks = TEST_SCENARIOS["multi_agent_tasks"]
        
        for task in tasks:
            start_time = time.time()
            
            # Execute multi-agent workflow
            agent_outputs = {}
            
            # Agent 1: Researcher
            research_prompt = f"As a research agent, analyze: {task['description']}. Focus on data gathering and market insights."
            research_response = await self.llm.ainvoke([HumanMessage(content=research_prompt)])
            agent_outputs["researcher"] = research_response.content
            
            # Agent 2: Analyst  
            analysis_prompt = f"""
            As an analyst, review this research: {research_response.content[:500]}...
            
            Provide analysis for: {task['description']}
            Focus on insights, trends, and implications.
            """
            analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            agent_outputs["analyst"] = analysis_response.content
            
            # Agent 3: Coordinator/Strategist
            coordination_prompt = f"""
            As a strategic coordinator, synthesize these inputs:
            
            Research: {research_response.content[:300]}...
            Analysis: {analysis_response.content[:300]}...
            
            Provide strategic recommendations for: {task['description']}
            """
            coordination_response = await self.llm.ainvoke([HumanMessage(content=coordination_prompt)])
            agent_outputs["coordinator"] = coordination_response.content
            
            end_time = time.time()
            
            # Evaluate success criteria
            final_output = coordination_response.content
            criteria_met = 0
            for criteria in task['success_criteria']:
                if any(keyword in final_output.lower() for keyword in criteria.lower().split()):
                    criteria_met += 1
            
            success_rate = criteria_met / len(task['success_criteria'])
            
            results.append({
                "task_id": task['task_id'],
                "description": task['description'],
                "agents_executed": len(agent_outputs),
                "success_criteria_met": criteria_met,
                "total_criteria": len(task['success_criteria']),
                "success_rate": success_rate,
                "success": success_rate >= 0.7,
                "total_latency_ms": (end_time - start_time) * 1000,
                "agent_outputs": {k: len(v) for k, v in agent_outputs.items()},  # Store lengths for size
                "coordination_quality": len(final_output)
            })
        
        state["results"].extend(results)
        state["current_step"] = "analysis"
        
        return state
    
    @traceable(name="generate_analysis")
    async def generate_analysis(self, state: EvaluationState) -> EvaluationState:
        """Generate comprehensive analysis and recommendations"""
        print("📊 Generating Analysis...")
        
        all_results = state["results"]
        
        # Categorize results
        conversation_results = [r for r in all_results if "scenario" in r]
        profile_results = [r for r in all_results if "user_id" in r]
        multiagent_results = [r for r in all_results if "task_id" in r]
        
        # Calculate overall metrics
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if r.get("success", True))
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Feature analysis
        feature_analysis = {
            "conversation_memory": {
                "test_count": len(conversation_results),
                "avg_success_rate": sum(r["success_rate"] for r in conversation_results) / len(conversation_results) if conversation_results else 0,
                "avg_latency": sum(r["avg_latency"] for r in conversation_results) / len(conversation_results) if conversation_results else 0
            },
            "user_profiles": {
                "test_count": len(profile_results),
                "avg_personalization": sum(r["personalization_score"] for r in profile_results) / len(profile_results) if profile_results else 0,
                "avg_latency": sum(r["latency_ms"] for r in profile_results) / len(profile_results) if profile_results else 0
            },
            "multi_agent": {
                "test_count": len(multiagent_results),
                "avg_success_rate": sum(r["success_rate"] for r in multiagent_results) / len(multiagent_results) if multiagent_results else 0,
                "avg_latency": sum(r["total_latency_ms"] for r in multiagent_results) / len(multiagent_results) if multiagent_results else 0
            }
        }
        
        # Generate recommendations
        recommendations = []
        if overall_success_rate >= 0.9:
            recommendations.append("✅ Excellent performance - Ready for production")
        elif overall_success_rate >= 0.7:
            recommendations.append("⚠️ Good performance - Minor optimizations recommended")
        else:
            recommendations.append("❌ Performance issues - Significant improvements needed")
        
        summary = {
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "feature_analysis": feature_analysis,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "langsmith_evidence": {
                "tracing_enabled": True,
                "real_time_monitoring": True,
                "multi_agent_coordination": len(multiagent_results) > 0,
                "session_management": len(conversation_results) > 0,
                "personalization": len(profile_results) > 0
            }
        }
        
        state["summary"] = summary
        state["current_step"] = "complete"
        
        return state

# ==================== GRAPH CONSTRUCTION ====================

def create_evaluation_graph():
    """Create the LangGraph evaluation workflow"""
    
    evaluator = LangSmithEvaluationGraph()
    
    # Define the graph
    workflow = StateGraph(EvaluationState)
    
    # Add nodes
    workflow.add_node("conversation_memory", evaluator.evaluate_conversation_memory)
    workflow.add_node("user_profiles", evaluator.evaluate_user_profiles) 
    workflow.add_node("multi_agent", evaluator.evaluate_multi_agent)
    workflow.add_node("analysis", evaluator.generate_analysis)
    
    # Define edges
    workflow.set_entry_point("conversation_memory")
    workflow.add_edge("conversation_memory", "user_profiles")
    workflow.add_edge("user_profiles", "multi_agent")
    workflow.add_edge("multi_agent", "analysis")
    workflow.add_edge("analysis", END)
    
    # Compile with checkpointer for state persistence
    checkpointer = SqliteSaver.from_conn_string(":memory:")
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

# ==================== CLOUD DEPLOYMENT ENTRY POINT ====================

@traceable(name="langsmith_evaluation_workflow")
async def run_evaluation(input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main entry point for LangGraph Cloud deployment"""
    
    # Initialize state
    initial_state: EvaluationState = {
        "test_type": input_data.get("test_type", "comprehensive") if input_data else "comprehensive",
        "test_data": input_data.get("test_data", {}) if input_data else {},
        "current_step": "conversation_memory",
        "results": [],
        "summary": {},
        "errors": [],
        "metadata": {
            "started_at": datetime.now().isoformat(),
            "deployment": "langgraph_cloud",
            "version": "1.0.0"
        }
    }
    
    # Create and run the evaluation graph
    app = create_evaluation_graph()
    
    try:
        # Execute the workflow
        final_state = await app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        
        # Return comprehensive results
        return {
            "success": True,
            "summary": final_state["summary"],
            "detailed_results": final_state["results"],
            "metadata": final_state["metadata"],
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "langgraph-cloud-evaluation"),
            "dashboard_url": f"https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'langgraph-cloud-evaluation')}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": initial_state["metadata"]
        }

# ==================== LOCAL TESTING COMPATIBILITY ====================

async def main():
    """For local testing - can be removed in cloud deployment"""
    print("🚀 LangSmith Evaluation - LangGraph Cloud Version")
    print("=" * 60)
    
    result = await run_evaluation()
    
    if result["success"]:
        print("✅ Evaluation completed successfully!")
        print(f"📊 Overall Success Rate: {result['summary']['overall_success_rate']:.1%}")
        print(f"🔗 Dashboard: {result['dashboard_url']}")
        
        # Save results
        filename = f"langgraph_cloud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"📋 Results saved to: {filename}")
    else:
        print(f"❌ Evaluation failed: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())