import { motion } from 'framer-motion'
import type { Peak } from '../../types/peak'
import { ArrowLeft, Wind, Thermometer, Gauge } from 'lucide-react'

interface PeakViewProps {
  peak: Peak
  onBack: () => void
}

export function PeakView({ peak, onBack }: PeakViewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-[var(--ds-bg)] p-6"
    >
      {/* Header */}
      <header className="max-w-7xl mx-auto flex items-center justify-between mb-8">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-4 py-2 text-sm text-[var(--ds-text-secondary)] hover:text-white transition-colors rounded-lg hover:bg-[var(--ds-bg-secondary)]"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Map
        </button>

        <div className="text-right">
          <h1 className="text-2xl font-bold text-white font-[var(--font-display)]">
            {peak.name}
          </h1>
          <p className="text-[var(--ds-cyan)] font-mono text-lg">
            {peak.heightM.toLocaleString()}m
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Mountain Display */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="relative aspect-[4/3] bg-gradient-to-br from-[var(--ds-bg-secondary)] to-[var(--ds-bg)] rounded-2xl border border-[var(--ds-bg-tertiary)] overflow-hidden"
        >
          {/* Placeholder mountain visualization */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-[var(--ds-cyan)]/10 border border-[var(--ds-cyan)]/20 flex items-center justify-center">
                <svg
                  viewBox="0 0 24 24"
                  className="w-12 h-12 text-[var(--ds-cyan)]"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path d="M3 17l6-6 4 4 8-8" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M10 20l4-12 4 12" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className="text-sm text-[var(--ds-text-muted)]">
                Mountain visualization
              </p>
              <p className="text-xs text-[var(--ds-text-muted)] mt-1">
                AI-generated image coming soon
              </p>
            </div>
          </div>

          {/* Weather cards */}
          <div className="absolute bottom-4 left-4 right-4 flex gap-3">
            <WeatherCard icon={<Wind className="w-4 h-4" />} label="Wind" value="45" unit="km/h" />
            <WeatherCard icon={<Thermometer className="w-4 h-4" />} label="Temp" value="-22" unit="°C" />
            <WeatherCard icon={<Gauge className="w-4 h-4" />} label="Pressure" value="612" unit="hPa" />
          </div>
        </motion.div>

        {/* Right: Prediction Form Placeholder */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="bg-[var(--ds-bg-secondary)] rounded-2xl border border-[var(--ds-bg-tertiary)] p-6"
        >
          <h2 className="text-lg font-semibold text-white mb-6 font-[var(--font-display)]">
            Summit Prediction
          </h2>

          <div className="space-y-6">
            {/* Step indicator */}
            <div className="flex items-center gap-2 mb-8">
              {[1, 2, 3].map((step) => (
                <div key={step} className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      step === 1
                        ? 'bg-[var(--ds-cyan)] text-[var(--ds-bg)]'
                        : 'bg-[var(--ds-bg-tertiary)] text-[var(--ds-text-muted)]'
                    }`}
                  >
                    {step}
                  </div>
                  {step < 3 && (
                    <div className="w-12 h-0.5 bg-[var(--ds-bg-tertiary)] mx-2" />
                  )}
                </div>
              ))}
            </div>

            {/* Route Selection Placeholder */}
            <div>
              <label className="block text-sm text-[var(--ds-text-secondary)] mb-3">
                Select Route
              </label>
              <div className="space-y-2">
                {peak.routes.map((route) => (
                  <button
                    key={route.id}
                    className="w-full p-4 text-left rounded-lg border border-[var(--ds-bg-tertiary)] hover:border-[var(--ds-cyan)]/50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium text-white">
                          {route.name}
                        </div>
                        <div className="text-xs text-[var(--ds-text-muted)]">
                          {route.difficulty} route
                        </div>
                      </div>
                      {route.historicalSuccessRate && (
                        <div className="text-right">
                          <div className="text-sm font-mono text-[var(--ds-cyan)]">
                            {Math.round(route.historicalSuccessRate * 100)}%
                          </div>
                          <div className="text-xs text-[var(--ds-text-muted)]">
                            historical
                          </div>
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Next button */}
            <button className="w-full py-3 px-4 bg-[var(--ds-cyan)] text-[var(--ds-bg)] font-medium rounded-lg hover:bg-[var(--ds-cyan)]/90 transition-colors">
              Continue to Date & Team
            </button>
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}

function WeatherCard({
  icon,
  label,
  value,
  unit,
}: {
  icon: React.ReactNode
  label: string
  value: string
  unit: string
}) {
  return (
    <div className="flex-1 bg-[var(--ds-bg)]/80 backdrop-blur-sm rounded-lg p-3 border border-[var(--ds-bg-tertiary)]">
      <div className="flex items-center gap-2 text-[var(--ds-text-muted)] mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className="font-mono">
        <span className="text-white">{value}</span>
        <span className="text-[var(--ds-text-muted)] text-sm ml-1">{unit}</span>
      </div>
    </div>
  )
}
