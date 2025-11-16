"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, Circle, Loader2, XCircle, ChevronRight } from 'lucide-react'
import { cn } from "@/lib/utils"

interface ReasoningPhase {
  name: string
  substeps?: string[]
  content: string
  confidence?: number
  status: "pending" | "active" | "complete" | "error"
  timestamp?: number
}

interface ReasoningDisplayProps {
  reasoning: {
    problem?: string
    phases?: ReasoningPhase[]
    answer?: string
    overall_confidence?: number
  } | null
  isActive: boolean
}

export function ReasoningDisplay({ reasoning, isActive }: ReasoningDisplayProps) {
  const [displayPhases, setDisplayPhases] = useState<ReasoningPhase[]>([])

  useEffect(() => {
    if (reasoning?.phases) {
      setDisplayPhases(reasoning.phases)
    }
  }, [reasoning])

  if (!reasoning && !isActive) {
    return (
      <Card className="flex min-h-[700px] items-center justify-center border-dashed border-border/50 bg-card/30 p-12">
        <div className="text-center">
          <div className="relative mx-auto mb-8 size-20">
            <div className="absolute inset-0 animate-pulse rounded-full bg-primary/10" />
            <Circle className="absolute inset-0 m-auto size-16 text-primary/30" strokeWidth={1} />
            <Circle className="absolute inset-0 m-auto size-12 text-primary/20" strokeWidth={1} />
          </div>
          <h3 className="text-lg font-semibold">Ready for Analysis</h3>
          <p className="mt-2 text-sm text-muted-foreground">Select a problem type and enter your question</p>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-5">
      {reasoning?.problem && (
        <Card className="group relative overflow-hidden border-primary/30 bg-gradient-to-br from-primary/5 to-transparent p-8 shadow-xl shadow-primary/5">
          <div className="absolute right-0 top-0 h-32 w-32 bg-primary/5 blur-3xl" />
          <div className="relative">
            <div className="mb-4 flex items-center gap-2">
              <div className="size-1.5 rounded-full bg-primary animate-pulse" />
              <span className="text-xs font-semibold uppercase tracking-widest text-primary">Problem Statement</span>
            </div>
            <p className="text-pretty font-mono text-base leading-relaxed">{reasoning.problem}</p>
          </div>
        </Card>
      )}

      {displayPhases.map((phase, index) => (
        <PhaseCard key={index} phase={phase} index={index} />
      ))}

      {reasoning?.answer && (
        <Card className="relative overflow-hidden border-primary/50 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-8 shadow-2xl shadow-primary/10">
          <div className="absolute right-0 top-0 h-40 w-40 bg-primary/10 blur-3xl" />
          <div className="relative">
            <div className="mb-6 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex size-12 items-center justify-center rounded-xl bg-primary/20">
                  <CheckCircle2 className="size-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-bold">Final Solution</h3>
                  <p className="text-xs text-muted-foreground">Comprehensive Analysis Complete</p>
                </div>
              </div>
              {reasoning.overall_confidence && (
                <Badge variant="outline" className="border-primary/50 bg-primary/10 font-mono text-sm text-primary">
                  {Math.round(reasoning.overall_confidence * 100)}% confident
                </Badge>
              )}
            </div>
            <div className="rounded-lg border border-primary/20 bg-background/50 p-6">
              <p className="text-pretty font-mono text-base leading-relaxed">{reasoning.answer}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

function PhaseCard({ phase, index }: { phase: ReasoningPhase; index: number }) {
  const [isAnimating, setIsAnimating] = useState(false)
  const [showSubsteps, setShowSubsteps] = useState(false)

  useEffect(() => {
    if (phase.status === "active") {
      setIsAnimating(true)
      setShowSubsteps(true)
      const timer = setTimeout(() => setIsAnimating(false), 1500)
      return () => clearTimeout(timer)
    } else if (phase.status === "complete") {
      setShowSubsteps(true)
    }
  }, [phase.status])

  const Icon =
    phase.status === "complete"
      ? CheckCircle2
      : phase.status === "active"
        ? Loader2
        : phase.status === "error"
          ? XCircle
          : Circle

  const phaseColors = [
    "from-cyan-500/10 to-blue-500/5 border-cyan-500/30",
    "from-blue-500/10 to-indigo-500/5 border-blue-500/30",
    "from-indigo-500/10 to-primary/5 border-indigo-500/30",
  ]

  return (
    <Card
      className={cn(
        "relative overflow-hidden border p-0 transition-all duration-700",
        phase.status === "active" && "shadow-2xl shadow-primary/30 ring-2 ring-primary/30 scale-[1.02]",
        phase.status === "complete" && `bg-gradient-to-br ${phaseColors[index % 3]}`,
        phase.status === "error" && "border-destructive/50 bg-destructive/5",
        phase.status === "pending" && "border-border/30 opacity-40",
      )}
    >
      {phase.status === "active" && (
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-primary to-transparent animate-scan opacity-50" />
        </div>
      )}

      <div className="p-6">
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className={cn(
                "flex size-14 items-center justify-center rounded-xl transition-all",
                phase.status === "complete" && "bg-primary/20",
                phase.status === "active" && "bg-primary/30 shadow-lg shadow-primary/20",
                phase.status === "pending" && "bg-muted/50",
                phase.status === "error" && "bg-destructive/20",
              )}
            >
              <Icon
                className={cn(
                  "size-7 transition-all",
                  phase.status === "complete" && "text-primary",
                  phase.status === "active" && "animate-spin text-primary",
                  phase.status === "pending" && "text-muted-foreground/30",
                  phase.status === "error" && "text-destructive",
                )}
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                  Phase {index + 1}
                </span>
                {phase.status === "active" && (
                  <div className="size-1.5 animate-pulse rounded-full bg-primary" />
                )}
              </div>
              <h3 className="mt-1 text-xl font-bold">{phase.name}</h3>
            </div>
          </div>

          {phase.confidence !== undefined && phase.status === "complete" && (
            <Badge variant="outline" className="font-mono text-sm">
              {Math.round(phase.confidence * 100)}%
            </Badge>
          )}
        </div>

        {phase.substeps && showSubsteps && phase.substeps.length > 0 && (
          <div className="mb-5 space-y-2 rounded-lg border border-border/30 bg-background/30 p-4">
            <div className="mb-3 flex items-center gap-2">
              <ChevronRight className="size-3 text-muted-foreground" />
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Analysis Steps
              </span>
            </div>
            <div className="space-y-2">
              {phase.substeps.map((substep, idx) => (
                <div key={idx} className="flex items-start gap-3 text-sm">
                  <div className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary/50" />
                  <span className="text-muted-foreground">{substep}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {phase.content && (
          <div className="rounded-lg border border-border/20 bg-background/50 p-5">
            <AnimatedText text={phase.content} isActive={phase.status === "active"} />
          </div>
        )}
      </div>
    </Card>
  )
}

function AnimatedText({ text, isActive }: { text: string; isActive: boolean }) {
  const [displayText, setDisplayText] = useState("")
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    if (isActive && text) {
      setDisplayText("")
      setCurrentIndex(0)
      let index = 0
      const interval = setInterval(() => {
        if (index <= text.length) {
          setDisplayText(text.slice(0, index))
          setCurrentIndex(index)
          index++
        } else {
          clearInterval(interval)
        }
      }, 12) // Slower for more deliberate, surgical feel
      return () => clearInterval(interval)
    } else {
      setDisplayText(text)
      setCurrentIndex(text.length)
    }
  }, [text, isActive])

  return (
    <div className="relative">
      <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground/90">
        {displayText}
        {isActive && currentIndex < text.length && (
          <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-primary" />
        )}
      </pre>
    </div>
  )
}
