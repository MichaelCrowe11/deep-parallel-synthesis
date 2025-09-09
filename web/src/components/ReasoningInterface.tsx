import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PlayIcon, 
  StopIcon, 
  CogIcon, 
  DocumentTextIcon,
  ChartBarIcon,
  EyeIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon
} from '@heroicons/react/24/outline';
import { useReasoningEngine } from '../hooks/useReasoningEngine';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { ReasoningChainVisualization } from './ReasoningChainVisualization';
import { ModelComparison } from './ModelComparison';
import { ProgressIndicator } from './ProgressIndicator';
import { ErrorBoundary } from './ErrorBoundary';
import { Button } from './ui/Button';
import { TextArea } from './ui/TextArea';
import { Select } from './ui/Select';
import { Toggle } from './ui/Toggle';
import { Card } from './ui/Card';

interface ReasoningInterfaceProps {
  className?: string;
}

export const ReasoningInterface: React.FC<ReasoningInterfaceProps> = ({ 
  className = '' 
}) => {
  const [prompt, setPrompt] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>(['gpt2']);
  const [showVisualization, setShowVisualization] = useState(true);
  const [showComparison, setShowComparison] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const { announceToScreenReader, highContrast, reducedMotion } = useAccessibility();
  
  const {
    reasoning,
    isLoading,
    error,
    executeReasoning,
    stopReasoning,
    clearResults
  } = useReasoningEngine();

  // Focus management
  useEffect(() => {
    if (!isLoading && reasoning?.response && autoSpeak) {
      speakText(reasoning.response);
    }
  }, [reasoning?.response, isLoading, autoSpeak]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!prompt.trim()) {
      announceToScreenReader('Please enter a prompt before submitting');
      promptRef.current?.focus();
      return;
    }

    announceToScreenReader('Starting reasoning process');
    
    try {
      await executeReasoning({
        prompt: prompt.trim(),
        models: selectedModels,
        visualize: showVisualization,
        compare: showComparison && selectedModels.length > 1
      });
      
      announceToScreenReader('Reasoning completed successfully');
    } catch (err) {
      announceToScreenReader('Reasoning failed. Please check the error details.');
    }
  };

  const speakText = (text: string) => {
    if ('speechSynthesis' in window && voiceEnabled) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      utterance.volume = 0.7;
      window.speechSynthesis.speak(utterance);
    }
  };

  const stopSpeech = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  };

  const reasoningTypes = [
    { value: 'deductive', label: 'Deductive Reasoning' },
    { value: 'inductive', label: 'Inductive Reasoning' },
    { value: 'abductive', label: 'Abductive Reasoning' },
    { value: 'causal', label: 'Causal Reasoning' },
    { value: 'analogical', label: 'Analogical Reasoning' },
    { value: 'systematic', label: 'Systematic Analysis' }
  ];

  const modelOptions = [
    { value: 'gpt2', label: 'GPT-2 (Fast)' },
    { value: 'llama3.1:8b', label: 'Llama 3.1 8B (Balanced)' },
    { value: 'llama3.1:70b', label: 'Llama 3.1 70B (Powerful)' },
    { value: 'deepparallel', label: 'DeepParallel (Specialized)' }
  ];

  return (
    <ErrorBoundary>
      <div className={`reasoning-interface ${className}`}>
        {/* Main Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Prompt Input */}
          <Card>
            <div className="p-6">
              <label 
                htmlFor="reasoning-prompt"
                className="block text-lg font-semibold text-gray-900 dark:text-white mb-3"
              >
                What would you like me to reason about?
              </label>
              
              <TextArea
                ref={promptRef}
                id="reasoning-prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your question or problem here. For example: 'Why do some materials conduct electricity better than others?' or 'What are the potential impacts of AI on healthcare?'"
                rows={4}
                className="w-full"
                disabled={isLoading}
                aria-describedby="prompt-help"
                required
              />
              
              <p id="prompt-help" className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                Be specific and clear in your question. The more context you provide, the better the reasoning will be.
              </p>
            </div>
          </Card>

          {/* Configuration Panel */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Reasoning Configuration
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Model Selection */}
                <div>
                  <label 
                    htmlFor="model-select"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                  >
                    AI Models
                  </label>
                  <Select
                    id="model-select"
                    multiple
                    value={selectedModels}
                    onChange={setSelectedModels}
                    options={modelOptions}
                    aria-describedby="model-help"
                  />
                  <p id="model-help" className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Select one or more models for reasoning
                  </p>
                </div>

                {/* Reasoning Type */}
                <div>
                  <label 
                    htmlFor="reasoning-type"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                  >
                    Reasoning Approach
                  </label>
                  <Select
                    id="reasoning-type"
                    defaultValue="systematic"
                    options={reasoningTypes}
                    aria-describedby="reasoning-help"
                  />
                  <p id="reasoning-help" className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Choose the type of reasoning approach
                  </p>
                </div>
              </div>

              {/* Toggle Options */}
              <div className="mt-6 space-y-4">
                <div className="flex flex-wrap gap-6">
                  <Toggle
                    label="Show Visualization"
                    checked={showVisualization}
                    onChange={setShowVisualization}
                    description="Display reasoning chains visually"
                  />
                  
                  <Toggle
                    label="Compare Models"
                    checked={showComparison}
                    onChange={setShowComparison}
                    disabled={selectedModels.length < 2}
                    description="Compare outputs from different models"
                  />
                  
                  <Toggle
                    label="Voice Output"
                    checked={voiceEnabled}
                    onChange={setVoiceEnabled}
                    description="Enable text-to-speech for results"
                  />
                  
                  {voiceEnabled && (
                    <Toggle
                      label="Auto-speak Results"
                      checked={autoSpeak}
                      onChange={setAutoSpeak}
                      description="Automatically read results aloud"
                    />
                  )}
                </div>
              </div>
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <div className="flex gap-3">
              <Button
                type="submit"
                disabled={isLoading || !prompt.trim()}
                className="inline-flex items-center"
                variant="primary"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <StopIcon className="w-5 h-5 mr-2" />
                    Stop Reasoning
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-5 h-5 mr-2" />
                    Start Reasoning
                  </>
                )}
              </Button>
              
              {reasoning && (
                <Button
                  type="button"
                  onClick={clearResults}
                  variant="outline"
                  size="lg"
                >
                  Clear Results
                </Button>
              )}
            </div>

            {voiceEnabled && reasoning?.response && (
              <div className="flex gap-2">
                <Button
                  type="button"
                  onClick={() => speakText(reasoning.response)}
                  variant="outline"
                  size="sm"
                  aria-label="Read results aloud"
                >
                  <SpeakerWaveIcon className="w-4 h-4" />
                </Button>
                
                <Button
                  type="button"
                  onClick={stopSpeech}
                  variant="outline"
                  size="sm"
                  aria-label="Stop reading"
                >
                  <SpeakerXMarkIcon className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>
        </form>

        {/* Progress Indicator */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-8"
            >
              <ProgressIndicator 
                stage="reasoning"
                progress={0.5}
                message="Analyzing parallel reasoning chains..."
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6"
            >
              <Card variant="error">
                <div className="p-4">
                  <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
                    Reasoning Error
                  </h3>
                  <p className="text-red-700 dark:text-red-300">
                    {error}
                  </p>
                  <Button
                    onClick={clearResults}
                    variant="outline"
                    size="sm"
                    className="mt-3"
                  >
                    Try Again
                  </Button>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {reasoning && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-8 space-y-6"
            >
              {/* Main Response */}
              <Card>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                      Reasoning Result
                    </h3>
                    <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                      <ChartBarIcon className="w-4 h-4" />
                      <span>Confidence: {(reasoning.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p>{reasoning.response}</p>
                  </div>
                  
                  {reasoning.evidence && reasoning.evidence.length > 0 && (
                    <div className="mt-6">
                      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        Supporting Evidence
                      </h4>
                      <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                        {reasoning.evidence.map((evidence, index) => (
                          <li key={index} className="flex items-start">
                            <DocumentTextIcon className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
                            {evidence}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </Card>

              {/* Visualization */}
              {showVisualization && reasoning.reasoning_chains && (
                <ReasoningChainVisualization 
                  chains={reasoning.reasoning_chains}
                  reducedMotion={reducedMotion}
                />
              )}

              {/* Model Comparison */}
              {showComparison && selectedModels.length > 1 && (
                <ModelComparison 
                  models={selectedModels}
                  prompt={prompt}
                />
              )}

              {/* Metrics */}
              <Card>
                <div className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Performance Metrics
                  </h3>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {reasoning.metrics.total_time?.toFixed(2) || 0}s
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Processing Time
                      </div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {reasoning.metrics.valid_chains || 0}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Valid Chains
                      </div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                        {(reasoning.metrics.synthesis_quality || 0).toFixed(2)}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Quality Score
                      </div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                        {(reasoning.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Confidence
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </ErrorBoundary>
  );
};