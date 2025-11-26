from .types import ComplianceInput, ComplianceResult

class ComplianceVerifier:
    def verify(self, input_data: ComplianceInput) -> ComplianceResult:
        # Check rules
        issues = []
        if input_data.beep_count > 5:
            issues.append("Too many beeps")
            
        # Check classification results
        if "hostile" in input_data.classification.behavior_tags:
            issues.append("Hostile behavior detected")
            
        if input_data.classification.status == "flagged":
            issues.append("Content flagged by classifier")
        
        return ComplianceResult(
            file_id=input_data.metadata.file_id,
            is_compliant=len(issues) == 0,
            issues=issues,
            details={
                "behavior": input_data.classification.behavior_tags,
                "duration": input_data.metadata.duration
            }
        )
