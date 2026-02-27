export interface Coordinates {
  lat: number
  lng: number
}

export interface Route {
  id: string
  name: string
  difficulty: 'Normal' | 'Technical' | 'Extreme'
  historicalSuccessRate?: number
}

export interface Peak {
  id: string
  name: string
  heightM: number
  country: string
  range: string
  coordinates: Coordinates
  routes: Route[]
  imageFolder: string
}

export type ViewState = 'map' | 'peak' | 'results'

export interface AppState {
  currentView: ViewState
  selectedPeakId: string | null
}
