"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, Circle, Loader2, XCircle, ChevronRight, Sparkles } from 'lucide-react'
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
          <div className="relative mx-auto mb-8 size-24">
            <div className="absolute inset-0 animate-ping rounded-full bg-primary/10 opacity-75" style={{ animationDuration: '3s' }} />
            <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
            <div className="absolute inset-3 rounded-full border-2 border-primary/10" />
            <Circle className="absolute inset-0 m-auto size-16 text-primary/40" strokeWidth={1.5} />
            <Circle className="absolute inset-0 m-auto size-10 text-primary/20" strokeWidth={1} />
          </div>
          <h3 className="text-xl font-bold">Ready for Surgical Analysis</h3>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground max-w-md mx-auto">
            The Crowe Logic Framework awaits. Select a problem domain and submit your query for methodical, multi-phase reasoning.
          </p>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {reasoning?.problem && (
        <Card className="group relative overflow-hidden border-primary/40 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-8 shadow-2xl shadow-primary/10">
          <div className="absolute -right-12 -top-12 size-40 rounded-full bg-primary/10 blur-3xl" />
          <div className="absolute -bottom-8 -left-8 size-32 rounded-full bg-accent/10 blur-2xl" />
          <div className="relative">
            <div className="mb-5 flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-primary/20">
                <Sparkles className="size-5 text-primary" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <div className="size-1.5 animate-pulse rounded-full bg-primary" />
                  <span className="text-xs font-bold uppercase tracking-widest text-primary">Problem Statement</span>
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">Input for surgical analysis</p>
              </div>
            </div>
            <div className="rounded-lg border border-primary/20 bg-background/50 p-6">
              <p className="text-pretty font-mono text-base leading-relaxed">{reasoning.problem}</p>
            </div>
          </div>
        </Card>
      )}

      {displayPhases.map((phase, index) => (
        <PhaseCard key={index} phase={phase} index={index} />
      ))}

      {reasoning?.answer && (
        <Card className="relative overflow-hidden border-primary/60 bg-gradient-to-br from-primary/15 via-primary/8 to-accent/5 p-10 shadow-2xl shadow-primary/20">
          <div className="absolute -right-16 -top-16 size-48 rounded-full bg-primary/15 blur-3xl" />
          <div className="absolute -bottom-12 -left-12 size-40 rounded-full bg-accent/10 blur-3xl" />
          <div className="relative">
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex size-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/30 to-primary/20 shadow-lg shadow-primary/30">
                  <CheckCircle2 className="size-8 text-primary" strokeWidth={2.5} />
                </div>
                <div>
                  <h3 className="text-2xl font-bold tracking-tight">Final Solution</h3>
                  <p className="mt-1 text-sm text-muted-foreground">Comprehensive multi-phase analysis complete</p>
                </div>
              </div>
              {reasoning.overall_confidence && (
                <Badge 
                  variant="outline" 
                  className="border-primary/60 bg-primary/20 px-4 py-2 font-mono text-base text-primary shadow-lg shadow-primary/20"
                >
                  {Math.round(reasoning.overall_confidence * 100)}% confident
                </Badge>
              )}
            </div>
            <div className="rounded-xl border-2 border-primary/30 bg-background/60 p-8 shadow-inner">
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
    { gradient: "from-cyan-500/15 via-blue-500/10 to-blue-500/5", border: "border-cyan-500/40", glow: "shadow-cyan-500/20" },
    { gradient: "from-blue-500/15 via-indigo-500/10 to-indigo-500/5", border: "border-blue-500/40", glow: "shadow-blue-500/20" },
    { gradient: "from-indigo-500/15 via-primary/10 to-primary/5", border: "border-indigo-500/40", glow: "shadow-indigo-500/20" },
  ]

  const colors = phaseColors[index % 3]

  return (
    <Card
      className={cn(
        "relative overflow-hidden border-2 p-0 transition-all duration-700",
        phase.status === "active" && `shadow-2xl ${colors.glow} ring-2 ring-primary/40 scale-[1.01]`,
        phase.status === "complete" && `bg-gradient-to-br ${colors.gradient} ${colors.border}`,
        phase.status === "error" && "border-destructive/50 bg-destructive/5",
        phase.status === "pending" && "border-border/30 opacity-50",
      )}
    >
      {phase.status === "active" && (
        <>
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="h-1 w-full bg-gradient-to-r from-transparent via-primary to-transparent animate-scan opacity-60" 
                 style={{ animationDuration: '2s' }} />
          </div>
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent animate-pulse" 
               style={{ animationDuration: '2s' }} />
        </>
      )}

      <div className="p-8">
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-5">
            <div
              className={cn(
                "flex size-16 items-center justify-center rounded-2xl transition-all duration-500",
                phase.status === "complete" && "bg-primary/25 shadow-lg shadow-primary/20",
                phase.status === "active" && "bg-primary/35 shadow-xl shadow-primary/30 scale-110",
                phase.status === "pending" && "bg-muted/50",
                phase.status === "error" && "bg-destructive/20",
              )}
            >
              <Icon
                className={cn(
                  "size-8 transition-all duration-500",
                  phase.status === "complete" && "text-primary",
                  phase.status === "active" && "animate-spin text-primary",
                  phase.status === "pending" && "text-muted-foreground/30",
                  phase.status === "error" && "text-destructive",
                )}
                strokeWidth={2.5}
              />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
                  Phase {index + 1}
                </span>
                {phase.status === "active" && (
                  <>
                    <div className="size-2 animate-pulse rounded-full bg-primary" />
                    <span className="text-xs font-semibold text-primary">Processing</span>
                  </>
                )}
              </div>
              <h3 className="mt-2 text-2xl font-bold tracking-tight">{phase.name}</h3>
            </div>
          </div>

          {phase.confidence !== undefined && phase.status === "complete" && (
            <Badge 
              variant="outline" 
              className="border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-sm shadow-sm"
            >
              {Math.round(phase.confidence * 100)}%
            </Badge>
          )}
        </div>

        {phase.substeps && showSubsteps && phase.substeps.length > 0 && (
          <div className="mb-6 space-y-2 rounded-xl border-2 border-border/40 bg-background/40 p-6 shadow-inner">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex size-7 items-center justify-center rounded-lg bg-primary/15">
                <ChevronRight className="size-4 text-primary" />
              </div>
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                Analysis Steps
              </span>
            </div>
            <div className="space-y-3">
              {phase.substeps.map((substep, idx) => (
                <div key={idx} className="flex items-start gap-4 text-sm">
                  <div className="mt-2 flex size-6 shrink-0 items-center justify-center rounded-md bg-primary/20">
                    <span className="text-xs font-bold text-primary">{idx + 1}</span>
                  </div>
                  <span className="flex-1 leading-relaxed text-muted-foreground">{substep}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {phase.content && (
          <div className="rounded-xl border-2 border-border/30 bg-background/60 p-6 shadow-inner">
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
      }, 8) // Faster animation for better UX
      return () => clearInterval(interval)
    } else {
      setDisplayText(text)
      setCurrentIndex(text.length)
    }
  }, [text, isActive])

  return (
    <div className="relative">
      <pre className="whitespace-pre-wrap font-mono text-sm leading-loose text-foreground/95">
        {displayText}
        {isActive && currentIndex < text.length && (
          <span className="ml-1 inline-block h-5 w-2.5 animate-pulse bg-primary shadow-lg shadow-primary/50" />
        )}
      </pre>
    </div>
  )
}
