import { generateObject } from "ai"
import { z } from "zod"
import { createClient } from "@/lib/supabase/server"
import { getCachedReasoning, setCachedReasoning } from "@/lib/redis/client"

const reasoningSchema = z.object({
  problem: z.string(),
  phases: z.array(
    z.object({
      name: z.string(),
      substeps: z.array(z.string()),
      content: z.string(),
      confidence: z.number().min(0).max(1),
      status: z.enum(["pending", "active", "complete", "error"]),
      timestamp: z.number().optional(),
    }),
  ),
  answer: z.string(),
  overall_confidence: z.number().min(0).max(1),
  knowledge_graph: z
    .object({
      nodes: z.array(
        z.object({
          id: z.string(),
          label: z.string(),
          type: z.string(),
        }),
      ),
      edges: z.array(
        z.object({
          from: z.string(),
          to: z.string(),
          label: z.string(),
        }),
      ),
    })
    .optional(),
})

const PROMPTS = {
  math: `You are an expert mathematical reasoner using the Crowe Logic Framework - a systematic three-phase methodology for deep problem analysis.

Break down this problem with surgical precision into THREE detailed phases:

**Phase 1: Context Analysis**
- Identify ALL variables, constants, and unknowns
- Parse mathematical relationships and constraints
- Map domain boundaries and edge cases
- Classify problem type and applicable theorems
Generate 4-6 substeps showing your analysis

**Phase 2: Logical Synthesis** 
- Apply relevant formulas and operations step-by-step
- Show intermediate calculations with full work
- Verify dimensional analysis and unit consistency
- Check for alternative solution paths
Generate 5-8 substeps showing each calculation

**Phase 3: Resolution**
- Synthesize final answer with full justification
- Cross-verify using alternative methods
- Validate answer against original constraints
- Assess confidence with uncertainty bounds
Generate 3-4 substeps showing verification

Create a knowledge graph showing mathematical dependencies between concepts.

Be thorough, methodical, and show EVERY step of your thinking.`,

  logic: `You are an expert logical reasoner using the Crowe Logic Framework for deep analytical thinking.

Break down this logic problem with precision into THREE detailed phases:

**Phase 1: Context Analysis**
- Parse all premises and conclusions
- Identify logical operators and quantifiers
- Map argument structure and dependencies
- Classify fallacies and validity conditions
Generate 4-6 substeps

**Phase 2: Logical Synthesis**
- Apply formal inference rules systematically
- Construct truth tables or proofs
- Evaluate validity using multiple methods
- Test counterexamples and edge cases
Generate 5-7 substeps

**Phase 3: Resolution**
- State final conclusion with formal justification
- Verify logical consistency
- Assess strength of argument
- Identify assumptions and limitations
Generate 3-5 substeps

Include a knowledge graph showing logical dependencies.

Be rigorous and show all reasoning steps.`,

  science: `You are an expert scientific reasoner using the Crowe Logic Framework for systematic analysis.

Break down this science problem with precision into THREE detailed phases:

**Phase 1: Context Analysis**
- Identify relevant scientific principles and laws
- Parse known and unknown variables
- Map cause-effect relationships
- Classify phenomena and mechanisms
Generate 4-6 substeps

**Phase 2: Logical Synthesis**
- Apply scientific principles step-by-step
- Show calculations or qualitative reasoning
- Connect micro and macro perspectives
- Consider alternative explanations
Generate 5-8 substeps

**Phase 3: Resolution**
- Provide comprehensive scientific explanation
- Verify against empirical evidence
- Assess confidence and limitations
- Suggest further investigation paths
Generate 3-5 substeps

Include knowledge graph connecting concepts, laws, and observations.

Be thorough and scientifically rigorous.`,

  code: `You are an expert coding reasoner using the Crowe Logic Framework for algorithmic thinking.

Break down this coding problem with precision into THREE detailed phases:

**Phase 1: Context Analysis**
- Parse requirements and constraints
- Identify inputs, outputs, and edge cases
- Analyze time/space complexity requirements
- Map problem to known patterns
Generate 4-6 substeps

**Phase 2: Logical Synthesis**
- Design algorithm with data structures
- Plan implementation with pseudocode
- Optimize for efficiency
- Handle edge cases systematically
Generate 6-10 substeps

**Phase 3: Resolution**
- Provide clean, working code
- Explain implementation choices
- Verify correctness with test cases
- Analyze final complexity
Generate 4-6 substeps

Include knowledge graph showing function dependencies and data flow.

Show detailed algorithmic thinking.`,
}

function generateMockReasoning(problem: string, type: string) {
  const phases = [
    {
      name: "Context Analysis",
      substeps: [
        "Parsing problem statement and identifying key elements",
        "Extracting variables, constants, and unknowns",
        "Mapping relationships and constraints",
        "Classifying problem type and applicable frameworks",
        "Assessing complexity and edge cases",
      ],
      content: `Deep analysis of problem structure:\n\n• Problem Domain: ${type}\n• Key components identified and catalogued\n• Constraint mapping completed\n• Applicable methodologies selected\n• Complexity assessment: moderate to high\n\nThis phase establishes the foundation for systematic reasoning by thoroughly understanding all problem dimensions.`,
      confidence: 0.87,
      status: "complete" as const,
      timestamp: Date.now(),
    },
    {
      name: "Logical Synthesis",
      substeps: [
        "Applying domain-specific principles",
        "Breaking down into sub-problems",
        "Executing step-by-step transformations",
        "Verifying intermediate results",
        "Checking for logical consistency",
        "Exploring alternative approaches",
        "Optimizing solution path",
      ],
      content: `Systematic application of reasoning principles:\n\n• Step 1: Foundation established using core principles\n• Step 2: Intermediate transformations applied\n• Step 3: Sub-problems solved methodically\n• Step 4: Results validated against constraints\n• Step 5: Alternative paths explored\n• Step 6: Optimal solution identified\n\nEach transformation builds upon previous steps with rigorous validation.`,
      confidence: 0.84,
      status: "complete" as const,
      timestamp: Date.now() + 2000,
    },
    {
      name: "Resolution",
      substeps: [
        "Synthesizing final answer from analysis",
        "Cross-verifying using alternative methods",
        "Validating against original constraints",
        "Assessing confidence with uncertainty bounds",
      ],
      content: `Final solution with comprehensive verification:\n\n• Primary solution: Derived through systematic analysis\n• Verification: Cross-checked using independent methods\n• Constraints: All original requirements satisfied\n• Confidence: High (87%) based on multi-path validation\n\nThe solution demonstrates logical consistency and empirical validity across all test conditions.`,
      confidence: 0.90,
      status: "complete" as const,
      timestamp: Date.now() + 4000,
    },
  ]

  const mockAnswer =
    type === "math"
      ? "Through systematic application of the Crowe Logic Framework, we arrive at a well-validated solution that satisfies all mathematical constraints with high confidence."
      : type === "code"
        ? "The implementation leverages optimal data structures and algorithms, achieving the required complexity bounds while handling all edge cases gracefully."
        : "The comprehensive analysis yields a well-supported conclusion grounded in rigorous logical principles and empirical evidence."

  return {
    problem,
    phases,
    answer: mockAnswer,
    overall_confidence: 0.87,
    knowledge_graph: {
      nodes: [
        { id: "problem", label: "Problem Input", type: "input" },
        { id: "context", label: "Context Analysis", type: "phase" },
        { id: "synthesis", label: "Logical Synthesis", type: "phase" },
        { id: "resolution", label: "Resolution", type: "phase" },
        { id: "solution", label: "Final Solution", type: "output" },
      ],
      edges: [
        { from: "problem", to: "context", label: "analyzes" },
        { from: "context", to: "synthesis", label: "informs" },
        { from: "synthesis", to: "resolution", label: "produces" },
        { from: "resolution", to: "solution", label: "yields" },
      ],
    },
    demo_mode: true,
  }
}

export async function POST(req: Request) {
  const { problem, type, userId } = await req.json()

  console.log("[v0] Reasoning request:", { problem, type })

  try {
    const cached = await getCachedReasoning(problem)
    if (cached) {
      console.log("[v0] Cache hit - returning cached reasoning")
      return Response.json({ ...cached, cached: true })
    }
  } catch (cacheError) {
    console.log("[v0] Cache check failed:", cacheError)
  }

  const hasGroq = !!process.env.GROQ_API_KEY
  const hasXAI = !!process.env.XAI_API_KEY
  const hasAIProvider = hasGroq || hasXAI

  const modelConfig = hasXAI
    ? { model: "xai/grok-beta", provider: "xAI" }
    : hasGroq
      ? { model: "groq/llama-3.3-70b-versatile", provider: "Groq" }
      : null

  console.log("[v0] AI Provider available:", hasAIProvider, modelConfig)

  if (!hasAIProvider || !modelConfig) {
    console.log("[v0] No AI provider, returning mock reasoning")
    const mockData = generateMockReasoning(problem, type)
    
    try {
      await setCachedReasoning(problem, mockData, 3600)
      const supabase = await createClient()
      await supabase.from("reasoning_sessions").insert({
        query: problem,
        response: JSON.stringify(mockData),
        model: "crowe-logic-v1",
        credits_used: 0,
        user_id: userId || null,
      })
    } catch (error) {
      console.log("[v0] Failed to cache/save mock data:", error)
    }
    
    return Response.json({ ...mockData, demo_mode: true })
  }

  try {
    console.log("[v0] Generating with AI Gateway:", modelConfig.model)
    
    const { generateObject } = await import("ai")
    
    const { object } = await generateObject({
      model: modelConfig.model,
      schema: reasoningSchema,
      prompt: `${PROMPTS[type as keyof typeof PROMPTS]}

Problem: ${problem}

Provide DETAILED step-by-step reasoning with multiple substeps for each phase. Be thorough and methodical.`,
      temperature: 0.3,
      maxTokens: 8000,
    })

    console.log("[v0] AI reasoning complete")

    try {
      await setCachedReasoning(problem, object, 3600)
      const supabase = await createClient()
      await supabase.from("reasoning_sessions").insert({
        query: problem,
        response: JSON.stringify(object),
        model: modelConfig.model,
        credits_used: 1,
        user_id: userId || null,
      })
    } catch (error) {
      console.log("[v0] Failed to cache/save AI result:", error)
    }

    return Response.json(object)
  } catch (aiError: any) {
    console.error("[v0] AI Gateway error:", aiError?.message || aiError)
    
    const mockData = generateMockReasoning(problem, type)
    return Response.json({ ...mockData, demo_mode: true, error: "AI Gateway error, using fallback reasoning" })
  }
}
