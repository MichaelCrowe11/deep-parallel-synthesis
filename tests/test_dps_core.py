"""
Test suite for DPS Core functionality
"""

import pytest
import asyncio
import json
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dps_core import DPSCore, ReasoningType, ReasoningChain


class TestDPSCore:
    """Test DPS Core functionality"""
    
    @pytest.fixture
    def dps_core(self):
        """Create a DPS Core instance for testing"""
        return DPSCore(
            num_chains=4,
            max_depth=3,
            synthesis_temperature=0.7,
            validation_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, dps_core):
        """Test DPS Core initialization"""
        assert dps_core.num_chains == 4
        assert dps_core.max_depth == 3
        assert dps_core.synthesis_temperature == 0.7
        assert dps_core.validation_threshold == 0.7
        assert len(dps_core.chains) == 0
    
    @pytest.mark.asyncio
    async def test_basic_reasoning(self, dps_core):
        """Test basic reasoning functionality"""
        prompt = "What causes rain?"
        result = await dps_core.reason(prompt)
        
        assert "response" in result
        assert "confidence" in result
        assert "reasoning_chains" in result
        assert "metrics" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_reasoning_with_context(self, dps_core):
        """Test reasoning with context"""
        prompt = "Explain the phenomenon"
        context = {"domain": "physics", "topic": "quantum mechanics"}
        
        result = await dps_core.reason(prompt, context=context)
        
        assert result is not None
        assert len(result["reasoning_chains"]) > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_type_selection(self, dps_core):
        """Test automatic reasoning type selection"""
        prompts_and_expected = [
            ("Why does water boil?", ReasoningType.CAUSAL),
            ("If all birds can fly and penguins are birds, then...", ReasoningType.DEDUCTIVE),
            ("Based on the pattern 2, 4, 8, 16...", ReasoningType.INDUCTIVE),
        ]
        
        for prompt, expected_type in prompts_and_expected:
            types = dps_core._select_reasoning_types(prompt)
            assert expected_type in types or ReasoningType.SYSTEMATIC in types
    
    def test_reasoning_chain(self):
        """Test ReasoningChain functionality"""
        chain = ReasoningChain(
            chain_id="test_chain",
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise="All humans are mortal"
        )
        
        chain.add_step("Socrates is human", confidence=0.95)
        chain.add_step("Therefore, Socrates is mortal", confidence=0.9)
        
        assert len(chain.steps) == 2
        assert chain.confidence > 0
        assert chain.confidence < 1
        
        chain_dict = chain.to_dict()
        assert chain_dict["chain_id"] == "test_chain"
        assert chain_dict["type"] == "deductive"
    
    def test_cross_pollination(self, dps_core):
        """Test cross-pollination between chains"""
        # Create test chains
        chain1 = ReasoningChain(
            chain_id="chain1",
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise="Test premise 1"
        )
        chain1.confidence = 0.8
        chain1.conclusions = ["Conclusion 1"]
        
        chain2 = ReasoningChain(
            chain_id="chain2",
            reasoning_type=ReasoningType.INDUCTIVE,
            premise="Test premise 2"
        )
        chain2.confidence = 0.85
        chain2.conclusions = ["Conclusion 2"]
        
        dps_core.chains = [chain1, chain2]
        dps_core._cross_pollinate()
        
        # Check that cross-pollination occurred
        assert "cross_pollination" in chain1.metadata or "cross_pollination" in chain2.metadata
    
    def test_validation(self, dps_core):
        """Test chain validation"""
        chain1 = ReasoningChain(
            chain_id="valid_chain",
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise="Valid premise"
        )
        chain1.confidence = 0.9
        
        chain2 = ReasoningChain(
            chain_id="invalid_chain",
            reasoning_type=ReasoningType.INDUCTIVE,
            premise="Invalid premise"
        )
        chain2.confidence = 0.3
        
        dps_core.chains = [chain1, chain2]
        valid_chains = dps_core._validate_chains()
        
        assert len(valid_chains) == 1
        assert valid_chains[0].chain_id == "valid_chain"
    
    def test_synthesis(self, dps_core):
        """Test result synthesis"""
        chain1 = ReasoningChain(
            chain_id="chain1",
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise="Premise 1"
        )
        chain1.confidence = 0.85
        chain1.conclusions = ["Conclusion 1"]
        chain1.evidence = ["Evidence 1"]
        
        chain2 = ReasoningChain(
            chain_id="chain2",
            reasoning_type=ReasoningType.INDUCTIVE,
            premise="Premise 2"
        )
        chain2.confidence = 0.80
        chain2.conclusions = ["Conclusion 2"]
        chain2.evidence = ["Evidence 2"]
        
        synthesis = dps_core._synthesize_results([chain1, chain2])
        
        assert "response" in synthesis
        assert "confidence" in synthesis
        assert "quality_score" in synthesis
        assert synthesis["num_chains"] == 2
        assert len(synthesis["reasoning_types"]) > 0
    
    def test_metrics(self, dps_core):
        """Test metrics collection"""
        initial_metrics = dps_core.get_metrics()
        assert "current_metrics" in initial_metrics
        assert "history_length" in initial_metrics
        assert initial_metrics["history_length"] == 0
    
    @pytest.mark.asyncio
    async def test_performance(self, dps_core):
        """Test performance metrics"""
        prompt = "Test prompt for performance"
        result = await dps_core.reason(prompt)
        
        metrics = result["metrics"]
        assert "total_time" in metrics
        assert "chains_created" in metrics
        assert "valid_chains" in metrics
        assert "avg_confidence" in metrics
        assert metrics["total_time"] > 0
    
    def test_clear_history(self, dps_core):
        """Test clearing history"""
        # Add some history
        dps_core.synthesis_history.append({"test": "data"})
        dps_core.performance_metrics["test"] = 1.0
        
        # Clear
        dps_core.clear_history()
        
        assert len(dps_core.synthesis_history) == 0
        assert len(dps_core.performance_metrics) == 0


class TestReasoningTypes:
    """Test reasoning type functionality"""
    
    def test_all_reasoning_types(self):
        """Test that all reasoning types are defined"""
        expected_types = [
            "DEDUCTIVE", "INDUCTIVE", "ABDUCTIVE", "CAUSAL",
            "PROBABILISTIC", "ANALOGICAL", "COUNTERFACTUAL",
            "SYSTEMATIC", "QUANTUM", "NEURAL"
        ]
        
        for type_name in expected_types:
            assert hasattr(ReasoningType, type_name)
    
    def test_reasoning_type_values(self):
        """Test reasoning type string values"""
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.QUANTUM.value == "quantum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])