# AI Integration Setup Guide

This Crowe Logic Reasoning System uses the Vercel AI Gateway for intelligent reasoning. To unlock full AI capabilities, you'll need to connect an AI provider integration.

## Current Status

✅ **Core Systems Active:**
- Crowe Logic Framework (3-phase reasoning)
- Redis caching layer
- Supabase database persistence
- Knowledge graph generation
- Training data pipeline

⚠️ **Demo Mode Active:**
The system is currently running with structured reasoning templates. To get real AI-powered reasoning with detailed calculations and code generation, connect an AI integration below.

## Setup Instructions

### Option 1: Use Vercel AI Gateway (Recommended)

The AI Gateway is already configured and will work automatically once you connect an AI provider.

**Supported Providers:**
- **OpenAI** - gpt-4o, gpt-4o-mini, gpt-4-turbo
- **Anthropic** - claude-3-5-sonnet, claude-3-opus
- **xAI (Grok)** - grok-beta
- **Groq** - llama models with ultra-fast inference

### Option 2: Direct API Keys

If you prefer direct API access, add these environment variables:

\`\`\`bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# xAI
XAI_API_KEY=xai-...

# Groq
GROQ_API_KEY=gsk_...
\`\`\`

## Connect an Integration

1. Open the **Connect** tab in the sidebar
2. Choose your preferred AI provider
3. Follow the connection prompts
4. Refresh and start reasoning!

## Benefits of AI Integration

Without AI (Demo Mode):
- Structured reasoning templates
- Generic analysis patterns
- Basic knowledge graphs

With AI (Full Mode):
- Deep mathematical calculations
- Code generation and debugging
- Scientific explanations with sources
- Complex logical proofs
- Adaptive reasoning strategies
- Confidence scoring based on actual inference

## Training Pipeline

Once connected, all reasoning sessions are automatically stored for fine-tuning:

\`\`\`bash
# Export training data
cd scripts/training
python data_exporter.py

# Train on GPU pod (10 hours recommended)
bash run_gpu_training.sh
\`\`\`

## Questions?

Check the sidebar for integration status and environment variables.
