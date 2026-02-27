import { useRef, useCallback, useState } from 'react'
import { motion } from 'framer-motion'
import Map, { Marker, type MapRef } from 'react-map-gl/mapbox'
import { PEAKS, searchPeaks } from '../../data/peaks'
import type { Peak } from '../../types/peak'
import { Search, Mountain } from 'lucide-react'
import { config } from '../../config'

const MAPBOX_TOKEN = config.mapbox.token

// Warn if token is missing
if (!MAPBOX_TOKEN) {
  console.warn('Mapbox token is not configured. Update src/config.ts with your token.')
}

interface MapViewProps {
  onSelectPeak: (peakId: string) => void
  selectedPeakId: string | null
}

export function MapView({ onSelectPeak, selectedPeakId }: MapViewProps) {
  const mapRef = useRef<MapRef>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Peak[]>([])
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [hoveredPeak, setHoveredPeak] = useState<string | null>(null)

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
    if (query.trim()) {
      const results = searchPeaks(query)
      setSearchResults(results)
      setShowSearchResults(true)
    } else {
      setSearchResults([])
      setShowSearchResults(false)
    }
  }, [])

  const handleSelectFromSearch = useCallback(
    (peak: Peak) => {
      setShowSearchResults(false)
      setSearchQuery('')

      // Fly to the peak location
      mapRef.current?.flyTo({
        center: [peak.coordinates.lng, peak.coordinates.lat],
        zoom: 7,
        pitch: 45,
        bearing: 20,
        duration: 2000,
      })

      onSelectPeak(peak.id)
    },
    [onSelectPeak]
  )

  const handleMarkerClick = useCallback(
    (peak: Peak) => {
      mapRef.current?.flyTo({
        center: [peak.coordinates.lng, peak.coordinates.lat],
        zoom: 7,
        pitch: 45,
        bearing: 20,
        duration: 2000,
      })

      onSelectPeak(peak.id)
    },
    [onSelectPeak]
  )

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="relative h-screen w-full overflow-hidden"
    >
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-20 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="flex items-center gap-3"
          >
            <div className="w-10 h-10 rounded-lg bg-[var(--ds-cyan)]/10 border border-[var(--ds-cyan)]/30 flex items-center justify-center">
              <Mountain className="w-5 h-5 text-[var(--ds-cyan)]" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white font-[var(--font-display)]">
                DeepSummit
              </h1>
              <p className="text-xs text-[var(--ds-text-muted)] tracking-wide uppercase">
                Summit Success Prediction
              </p>
            </div>
          </motion.div>

          {/* Search */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="relative"
          >
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--ds-text-muted)]" />
              <input
                type="text"
                placeholder="Search peaks..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                onFocus={() => searchQuery && setShowSearchResults(true)}
                onBlur={() => setTimeout(() => setShowSearchResults(false), 200)}
                className="w-64 pl-10 pr-4 py-2.5 bg-[var(--ds-bg-secondary)]/80 backdrop-blur-sm border border-[var(--ds-bg-tertiary)] rounded-lg text-sm text-white placeholder:text-[var(--ds-text-muted)] focus:outline-none focus:border-[var(--ds-cyan)]/50 focus:ring-1 focus:ring-[var(--ds-cyan)]/20 transition-all"
              />
            </div>

            {/* Search Results Dropdown */}
            {showSearchResults && searchResults.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute top-full mt-2 w-full bg-[var(--ds-bg-secondary)] border border-[var(--ds-bg-tertiary)] rounded-lg overflow-hidden shadow-xl"
              >
                {searchResults.map((peak) => (
                  <button
                    key={peak.id}
                    onClick={() => handleSelectFromSearch(peak)}
                    className="w-full px-4 py-3 text-left hover:bg-[var(--ds-bg-tertiary)] transition-colors"
                  >
                    <div className="text-sm font-medium text-white">
                      {peak.name}
                    </div>
                    <div className="text-xs text-[var(--ds-text-muted)]">
                      {peak.heightM.toLocaleString()}m · {peak.country}
                    </div>
                  </button>
                ))}
              </motion.div>
            )}
          </motion.div>
        </div>
      </header>

      {/* Map Container - absolute positioning ensures proper dimensions */}
      <div className="absolute inset-0">
        <Map
          ref={mapRef}
          mapboxAccessToken={MAPBOX_TOKEN}
          initialViewState={{
            longitude: 84.5,
            latitude: 30,
            zoom: 4,
            pitch: 0,
            bearing: 0,
          }}
          style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
          mapStyle="mapbox://styles/mapbox/dark-v11"
          attributionControl={false}
        >
        {PEAKS.map((peak) => (
          <Marker
            key={peak.id}
            longitude={peak.coordinates.lng}
            latitude={peak.coordinates.lat}
            anchor="center"
          >
            <button
              onClick={() => handleMarkerClick(peak)}
              onMouseEnter={() => setHoveredPeak(peak.id)}
              onMouseLeave={() => setHoveredPeak(null)}
              className="group relative cursor-pointer"
            >
              {/* Glow effect */}
              <div
                className={`absolute inset-0 rounded-full bg-[var(--ds-cyan)] blur-md transition-opacity duration-300 ${
                  hoveredPeak === peak.id || selectedPeakId === peak.id
                    ? 'opacity-60'
                    : 'opacity-30'
                }`}
              />
              {/* Marker dot */}
              <div
                className={`relative w-3 h-3 rounded-full bg-[var(--ds-cyan)] transition-transform duration-300 ${
                  hoveredPeak === peak.id || selectedPeakId === peak.id
                    ? 'scale-150'
                    : 'scale-100'
                } animate-pulse-glow`}
              />
              {/* Tooltip */}
              {hoveredPeak === peak.id && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 px-3 py-2 bg-[var(--ds-bg-secondary)] border border-[var(--ds-bg-tertiary)] rounded-lg whitespace-nowrap shadow-xl"
                >
                  <div className="text-sm font-medium text-white">
                    {peak.name}
                  </div>
                  <div className="text-xs text-[var(--ds-cyan)] font-mono">
                    {peak.heightM.toLocaleString()}m
                  </div>
                </motion.div>
              )}
            </button>
          </Marker>
        ))}
      </Map>
      </div>

      {/* Bottom gradient overlay */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[var(--ds-bg)] to-transparent pointer-events-none" />

      {/* Footer info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.6 }}
        className="absolute bottom-6 left-1/2 -translate-x-1/2 text-center"
      >
        <p className="text-sm text-[var(--ds-text-secondary)]">
          Select a peak to begin your summit prediction
        </p>
        <p className="text-xs text-[var(--ds-text-muted)] mt-1">
          {PEAKS.length} eight-thousanders · AI-powered success analysis
        </p>
      </motion.div>
    </motion.div>
  )
}
