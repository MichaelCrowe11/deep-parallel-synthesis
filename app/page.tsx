"use client"

import { useState, useEffect } from "react"
import { Brain, Code2, FlaskConical, Calculator, Sparkles, ChevronRight } from 'lucide-react'
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
      console.log("[v0] Starting reasoning request:", { problem, type: selectedType })

      const response = await fetch("/api/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem, type: selectedType }),
      })

      console.log("[v0] Response status:", response.status)

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`)
      }

      const contentType = response.headers.get("content-type")
      
      if (contentType?.includes("application/json")) {
        // Non-streaming response (cached or mock)
        const data = await response.json()
        console.log("[v0] Received JSON response")
        if (data.demo_mode) setShowDemoNote(true)
        if (data.cached) console.log("[v0] Using cached result")
        setReasoning(data)
      } else if (contentType?.includes("text/plain")) {
        // AI SDK streamObject returns text/plain with special format
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error("Response body is not readable")
        }

        let buffer = ""
        let latestData: any = null

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            console.log("[v0] Received data chunk")
            buffer += decoder.decode(value, { stream: true })
            
            // Split by newlines to process complete lines
            const lines = buffer.split("\n")
            buffer = lines.pop() || "" // Keep incomplete line in buffer

            for (const line of lines) {
              if (!line.trim()) continue

              // AI SDK format: lines start with "0:" followed by JSON
              if (line.startsWith("0:")) {
                try {
                  const jsonStr = line.slice(2) // Remove "0:" prefix
                  latestData = JSON.parse(jsonStr)
                  setReasoning(latestData)
                  console.log("[v0] Parsed stream update")
                } catch (parseError) {
                  console.error("[v0] Parse error:", parseError, "Line:", line.slice(0, 100))
                }
              }
              // Handle error lines
              else if (line.startsWith("3:")) {
                const errorStr = line.slice(2)
                console.error("[v0] Stream error:", errorStr)
                throw new Error(`Stream error: ${errorStr}`)
              }
            }
          }

          // Process any remaining buffer
          if (buffer.trim() && buffer.startsWith("0:")) {
            try {
              const jsonStr = buffer.slice(2)
              latestData = JSON.parse(jsonStr)
              setReasoning(latestData)
            } catch (e) {
              console.error("[v0] Final buffer parse error:", e)
            }
          }

          // Final update with complete data
          if (latestData) {
            setReasoning(latestData)
          }
        } catch (streamError) {
          console.error("[v0] Stream processing error:", streamError)
          throw streamError
        }
      } else {
        throw new Error(`Unexpected content type: ${contentType}`)
      }
    } catch (error: any) {
      console.error("[v0] Reasoning error:", error)
      const errorMessage = error?.message || "An unexpected error occurred"
      setError(errorMessage)
    } finally {
      setIsReasoning(false)
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="relative overflow-hidden border-b border-border/50">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-background to-background" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent" />

        <div className="relative mx-auto max-w-7xl px-6 py-20 sm:py-28">
          <div className="mx-auto max-w-3xl">
            <div className="mb-12 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/20">
                  <Brain className="size-7 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-bold tracking-tight">Crowe Logic</h2>
                  <p className="text-xs text-muted-foreground">Reasoning Framework</p>
                </div>
              </div>
              <Badge variant="outline" className="border-primary/30 bg-primary/5 text-primary">
                v1.0
              </Badge>
            </div>

            <h1 className="text-pretty text-5xl font-bold tracking-tight sm:text-6xl lg:text-7xl">
              Surgical
              <span className="text-primary"> Reasoning</span>
            </h1>

            <p className="text-balance mt-6 text-xl leading-relaxed text-muted-foreground">
              Watch AI dissect problems with methodical precision. Experience deep, step-by-step analysis across
              mathematics, logic, science, and code.
            </p>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-6 py-12">
        {showDemoNote && (
          <Card className="mb-8 border-primary/30 bg-gradient-to-r from-primary/5 to-transparent p-6">
            <div className="flex items-start gap-4">
              <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                <Sparkles className="size-5 text-primary" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">Demo Mode Active</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Currently using structured reasoning templates. Connect an AI provider (xAI, Groq, OpenAI) for full
                  AI-powered analysis.
                </p>
              </div>
            </div>
          </Card>
        )}

        {error && (
          <Card className="mb-8 border-destructive/50 bg-destructive/5 p-6">
            <p className="text-sm text-destructive">
              <strong>Error:</strong> {error}
            </p>
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
