from pydantic import BaseModel, Field
from typing import List, Any

class SemanticFinding(BaseModel):
    requirement_id: str = Field(description="The R-index of the audited requirement.")
    verification: str = Field(description="Must be either 'cleared', 'partially_cleared', or 'failed'.")
    issue_type: str = Field(description="Classification: 'compliant', 'contradiction', 'omission', or 'additional'.")
    policy_citation: str = Field(description="Strict citation format: [Policy: Page X, Section Y, on Topic]")
    user_citation: str = Field(description="Strict citation format: [User Document: Page X, Section Y, on Topic] or 'Not present'")
    review_highlight: str = Field(description="Detailed explanation of semantic variance or gaps.")
    suggested_edits: str = Field(description="Exact rewrite or addition instructions to resolve the gap.")

class ComplianceReportSchema(BaseModel):
    policy_version: str = Field(default="v1_semantic", description="Version of the policy check")
    issue_count: int = Field(description="Total number of failed or partially-cleared or non-compliant issues found")
    findings: List[SemanticFinding] = Field(description="List of all evaluated requirements")