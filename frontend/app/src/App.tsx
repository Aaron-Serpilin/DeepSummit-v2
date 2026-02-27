import { useState, useCallback } from 'react'
import { AnimatePresence } from 'framer-motion'
import type { ViewState } from './types/peak'
import { getPeakById } from './data/peaks'
import { MapView } from './components/pages/MapView'
import { PeakView } from './components/pages/PeakView'

function App() {
  const [currentView, setCurrentView] = useState<ViewState>('map')
  const [selectedPeakId, setSelectedPeakId] = useState<string | null>(null)

  const selectedPeak = selectedPeakId ? getPeakById(selectedPeakId) : null

  const handleSelectPeak = useCallback((peakId: string) => {
    setSelectedPeakId(peakId)
    // Small delay to allow map fly-to animation to progress
    setTimeout(() => {
      setCurrentView('peak')
    }, 1500)
  }, [])

  const handleBackToMap = useCallback(() => {
    setCurrentView('map')
    setSelectedPeakId(null)
  }, [])

  return (
    <div className="relative min-h-screen bg-[var(--ds-bg)] noise-overlay">
      <AnimatePresence mode="wait">
        {currentView === 'map' && (
          <MapView
            key="map"
            onSelectPeak={handleSelectPeak}
            selectedPeakId={selectedPeakId}
          />
        )}
        {currentView === 'peak' && selectedPeak && (
          <PeakView
            key="peak"
            peak={selectedPeak}
            onBack={handleBackToMap}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
