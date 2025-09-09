"""
Test suite for DPS Validator functionality
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dps.validator import (
    ScientificValidator,
    LogicalValidator,
    MathematicalValidator,
    ValidationStatus,
    ValidationCategory,
    ValidationResult
)


class TestLogicalValidator:
    """Test logical validation functionality"""
    
    @pytest.fixture
    def validator(self):
        return LogicalValidator()
    
    def test_non_contradiction(self, validator):
        """Test detection of logical contradictions"""
        # No contradiction
        props1 = ["A is true", "B is true", "C follows from A and B"]
        result1 = validator.validate(props1)
        assert result1.metrics["contradiction_free"] == 1.0
        
        # With contradiction
        props2 = ["A is true", "not A is true", "B is true"]
        result2 = validator.validate(props2)
        assert result2.metrics["contradiction_free"] == 0.0
    
    def test_modus_ponens(self, validator):
        """Test modus ponens inference"""
        conclusion = validator._check_modus_ponens([
            "A implies B",
            "A"
        ])
        assert conclusion == "B"
    
    def test_modus_tollens(self, validator):
        """Test modus tollens inference"""
        conclusion = validator._check_modus_tollens([
            "A implies B",
            "not B"
        ])
        assert conclusion == "not A"
    
    def test_syllogism(self, validator):
        """Test syllogistic reasoning"""
        conclusion = validator._check_syllogism([
            "All humans are mortal",
            "All Greeks are humans"
        ])
        assert conclusion == "All Greeks are mortal"
    
    def test_transitivity(self, validator):
        """Test transitive relations"""
        conclusions = validator._check_transitivity([
            "A > B",
            "B > C"
        ])
        assert "A > C" in conclusions
    
    def test_validation_result(self, validator):
        """Test validation result structure"""
        props = ["If A then B", "A", "Therefore B"]
        result = validator.validate(props)
        
        assert isinstance(result, ValidationResult)
        assert result.category == ValidationCategory.LOGICAL_CONSISTENCY
        assert 0 <= result.confidence <= 1
        assert hasattr(result, "is_valid")


class TestMathematicalValidator:
    """Test mathematical validation functionality"""
    
    @pytest.fixture
    def validator(self):
        return MathematicalValidator()
    
    def test_simple_expression(self, validator):
        """Test validation of simple mathematical expressions"""
        result = validator.validate("2 + 2", expected_result=4)
        assert result.is_valid()
        assert result.confidence > 0.8
    
    def test_algebraic_expression(self, validator):
        """Test validation of algebraic expressions"""
        result = validator.validate("x**2 + 2*x + 1")
        assert result.metrics["parse_success"] == 1.0
    
    def test_invalid_expression(self, validator):
        """Test handling of invalid expressions"""
        result = validator.validate("2 + + 3")
        assert not result.is_valid()
        assert result.status == ValidationStatus.INVALID
    
    def test_result_checking(self, validator):
        """Test checking expected results"""
        result = validator.validate("3 * 4", expected_result=12)
        assert result.is_valid()
        
        result2 = validator.validate("3 * 4", expected_result=11)
        assert not result2.is_valid()


class TestScientificValidator:
    """Test scientific validation functionality"""
    
    @pytest.fixture
    def validator(self):
        return ScientificValidator()
    
    def test_complete_validation(self, validator):
        """Test complete scientific validation"""
        content = """
        If all mammals are warm-blooded and whales are mammals,
        then whales are warm-blooded. This can be expressed as:
        A → B, where A represents being a mammal.
        Studies show that whale body temperature is 36°C.
        """
        
        result = validator.validate(
            content=content,
            reasoning_type="deductive",
            evidence=["Marine Biology Journal, 2020", "Whale Studies, 2019"]
        )
        
        assert isinstance(result, ValidationResult)
        assert result.confidence > 0
    
    def test_empirical_validation(self, validator):
        """Test empirical evidence validation"""
        evidence = [
            "Smith et al., 2020",
            "Data from experiment X",
            "Peer-reviewed study in Nature"
        ]
        
        result = validator._validate_empirical_support(evidence)
        assert result.category == ValidationCategory.EMPIRICAL_SUPPORT
        assert result.confidence > 0.5
    
    def test_theoretical_alignment(self, validator):
        """Test theoretical alignment validation"""
        deductive_content = "If A then B. A is true. Therefore B."
        result = validator._validate_theoretical_alignment(
            deductive_content,
            "deductive"
        )
        assert result.confidence > 0.7
        
        inductive_content = "Observed pattern in data suggests trend."
        result2 = validator._validate_theoretical_alignment(
            inductive_content,
            "inductive"
        )
        assert result2.confidence > 0.5
    
    def test_extract_propositions(self, validator):
        """Test extraction of logical propositions"""
        content = """
        If the temperature rises, then ice melts.
        The temperature is rising.
        Therefore, ice will melt.
        All ice is frozen water.
        """
        
        props = validator._extract_logical_propositions(content)
        assert len(props) > 0
        assert any("if" in p.lower() for p in props)
        assert any("therefore" in p.lower() for p in props)
    
    def test_extract_math(self, validator):
        """Test extraction of mathematical expressions"""
        content = """
        The equation is x = 2y + 3.
        We can also write $E = mc^2$.
        The result equals 42.
        """
        
        expressions = validator._extract_mathematical_expressions(content)
        assert len(expressions) > 0
    
    def test_validation_summary(self, validator):
        """Test validation summary generation"""
        # Perform some validations
        validator.validate("Test content 1", "deductive")
        validator.validate("Test content 2", "inductive")
        
        summary = validator.get_validation_summary()
        assert "total_validations" in summary
        assert "average_confidence" in summary
        assert summary["total_validations"] == 2


class TestValidationStatus:
    """Test validation status enum"""
    
    def test_all_statuses(self):
        """Test all validation statuses are defined"""
        expected = ["VALID", "INVALID", "PARTIALLY_VALID", "UNCERTAIN", "REQUIRES_REVIEW"]
        for status in expected:
            assert hasattr(ValidationStatus, status)
    
    def test_status_values(self):
        """Test status string values"""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.INVALID.value == "invalid"


class TestValidationCategory:
    """Test validation category enum"""
    
    def test_all_categories(self):
        """Test all validation categories are defined"""
        expected = [
            "LOGICAL_CONSISTENCY",
            "MATHEMATICAL_CORRECTNESS",
            "EMPIRICAL_SUPPORT",
            "THEORETICAL_ALIGNMENT"
        ]
        for cat in expected:
            assert hasattr(ValidationCategory, cat)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])