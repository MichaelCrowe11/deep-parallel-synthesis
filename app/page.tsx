"use client"

import { useState, useEffect } from "react"
import { Brain, Code2, FlaskConical, Calculator, Sparkles, ChevronRight, XCircle } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ReasoningDisplay } from "@/components/reasoning-display"
import { KnowledgeGraph } from "@/components/knowledge-graph"

const PROBLEM_TYPES = [
  { id: "math", label: "Mathematics", icon: Calculator, color: "cyan" },
  { id: "logic", label: "Logic", icon: Brain, color: "blue" },
  { id: "science", label: "Science", icon: FlaskConical, color: "teal" },
  { id: "code", label: "Code", icon: Code2, color: "indigo" },
]

const EXAMPLE_PROBLEMS = {
  math: "If a train travels at 60 mph for 2.5 hours, then increases speed to 80 mph for 1.5 hours, what's the total distance traveled?",
  logic: "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
  science: "Why does ice float on water despite being solid?",
  code: "Write a function to find the longest palindrome substring in a given string.",
}

export default function Page() {
  const [selectedType, setSelectedType] = useState<string>("math")
  const [problem, setProblem] = useState("")
  const [isReasoning, setIsReasoning] = useState(false)
  const [reasoning, setReasoning] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [showDemoNote, setShowDemoNote] = useState(false)
  const [showKnowledgeGraph, setShowKnowledgeGraph] = useState(false)

  const handleSolve = async () => {
    if (!problem.trim()) return

    setIsReasoning(true)
    setReasoning(null)
    setError(null)
    setShowDemoNote(false)

    try {
      const response = await fetch("/api/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem, type: selectedType }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.demo_mode === true) {
        setShowDemoNote(true)
      }
      
      setReasoning(data)
    } catch (error: any) {
      console.error("[v0] Reasoning error:", error)
      setError(error?.message || "An unexpected error occurred")
    } finally {
      setIsReasoning(false)
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="relative overflow-hidden border-b border-border/50">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/8 via-background to-background" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/15 via-transparent to-transparent" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-accent/10 via-transparent to-transparent" />

        <div className="relative mx-auto max-w-7xl px-6 py-20 sm:py-32">
          <div className="mx-auto max-w-3xl">
            <div className="mb-16 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/10 ring-2 ring-primary/30 shadow-lg shadow-primary/20">
                  <Brain className="size-8 text-primary" strokeWidth={2} />
                </div>
                <div>
                  <h2 className="text-2xl font-bold tracking-tight">Crowe Logic</h2>
                  <p className="text-xs tracking-wide text-muted-foreground">Surgical Reasoning Framework</p>
                </div>
              </div>
              <Badge variant="outline" className="border-primary/40 bg-primary/10 px-4 py-1.5 text-primary shadow-sm">
                v1.0
              </Badge>
            </div>

            <h1 className="text-pretty text-5xl font-bold tracking-tight sm:text-6xl lg:text-7xl">
              Surgical
              <span className="bg-gradient-to-r from-primary via-primary to-accent bg-clip-text text-transparent"> Precision</span> Reasoning
            </h1>

            <p className="text-balance mt-8 text-xl leading-relaxed text-muted-foreground">
              Watch AI dissect complex problems with methodical precision. Experience deep, multi-phase analysis across
              mathematics, logic, science, and code.
            </p>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-6 py-12">
        {showDemoNote && (
          <Card className="mb-8 border-amber-500/30 bg-gradient-to-r from-amber-500/10 to-transparent p-6">
            <div className="flex items-start gap-4">
              <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-amber-500/20">
                <Sparkles className="size-5 text-amber-500" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-amber-500">Demo Mode Active</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Using structured reasoning templates. For full AI-powered analysis, ensure your API keys are properly configured.
                </p>
              </div>
            </div>
          </Card>
        )}

        {error && (
          <Card className="mb-8 border-destructive/50 bg-destructive/10 p-6">
            <div className="flex items-start gap-4">
              <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-destructive/20">
                <XCircle className="size-5 text-destructive" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-destructive">Error</h3>
                <p className="mt-1 text-sm text-destructive/90">{error}</p>
              </div>
            </div>
          </Card>
        )}

        <div className="grid gap-8 lg:grid-cols-[380px_1fr]">
          <div className="space-y-6">
            <Card className="border-border/50 bg-card/50 p-6 backdrop-blur-sm">
              <h3 className="mb-5 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                Problem Type
              </h3>

              <div className="space-y-3">
                {PROBLEM_TYPES.map((type) => {
                  const Icon = type.icon
                  const isSelected = selectedType === type.id
                  return (
                    <button
                      key={type.id}
                      onClick={() => {
                        setSelectedType(type.id)
                        setProblem(EXAMPLE_PROBLEMS[type.id as keyof typeof EXAMPLE_PROBLEMS])
                      }}
                      className={`group flex w-full items-center gap-4 rounded-lg border p-4 transition-all ${
                        isSelected
                          ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
                          : "border-border/50 hover:border-primary/50 hover:bg-card"
                      }`}
                    >
                      <div
                        className={`flex size-10 items-center justify-center rounded-lg transition-all ${
                          isSelected ? "bg-primary/20" : "bg-muted group-hover:bg-primary/10"
                        }`}
                      >
                        <Icon className={`size-5 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                      </div>
                      <span className={`font-medium ${isSelected ? "text-primary" : ""}`}>{type.label}</span>
                      {isSelected && <ChevronRight className="ml-auto size-4 text-primary" />}
                    </button>
                  )
                })}
              </div>
            </Card>

            <Card className="border-border/50 bg-card/50 p-6 backdrop-blur-sm">
              <h3 className="mb-5 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                Your Problem
              </h3>

              <Textarea
                value={problem}
                onChange={(e) => setProblem(e.target.value)}
                placeholder="Enter your problem here..."
                className="min-h-[180px] resize-none border-border/50 bg-background/50 font-mono text-sm"
              />

              <Button
                onClick={handleSolve}
                disabled={!problem.trim() || isReasoning}
                className="mt-4 w-full bg-primary hover:bg-primary/90"
                size="lg"
              >
                {isReasoning ? (
                  <>
                    <span className="relative mr-2 flex size-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary-foreground opacity-75" />
                      <span className="relative inline-flex size-2 rounded-full bg-primary-foreground" />
                    </span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 size-4" />
                    Begin Analysis
                  </>
                )}
              </Button>

              {reasoning?.knowledge_graph && (
                <Button
                  onClick={() => setShowKnowledgeGraph(!showKnowledgeGraph)}
                  variant="outline"
                  className="mt-3 w-full"
                  size="sm"
                >
                  {showKnowledgeGraph ? "Hide" : "Show"} Knowledge Graph
                </Button>
              )}
            </Card>
          </div>

          <div className="space-y-6">
            {showKnowledgeGraph && reasoning?.knowledge_graph && (
              <KnowledgeGraph
                nodes={reasoning.knowledge_graph.nodes || []}
                edges={reasoning.knowledge_graph.edges || []}
              />
            )}
            
            <ReasoningDisplay reasoning={reasoning} isActive={isReasoning} />
          </div>
        </div>
      </div>
    </main>
  )
}
