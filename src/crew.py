from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.schemas import ComplianceReportSchema
# Assuming you have your tools or knowledge sources imported here

@CrewBase
class ComplianceCrew():
    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    def __init__(self, knowledge_sources=None):
        self.knowledge_sources = knowledge_sources or []

    @agent
    def semantic_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['semantic_evaluator'],
            knowledge_sources=self.knowledge_sources, # Grants access to the policy PDF
            verbose=True
        )

    @task
    def semantic_evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['semantic_evaluation_task'],
            agent=self.semantic_evaluator(),
            # This is the magic line that replaces the LangChain parser:
            output_pydantic=ComplianceReportSchema 
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )