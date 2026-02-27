import type { Peak } from '../types/peak'

export const PEAKS: Peak[] = [
  {
    id: 'everest',
    name: 'Mount Everest',
    heightM: 8849,
    country: 'Nepal/China',
    range: 'Mahalangur Himal',
    coordinates: { lat: 27.9881, lng: 86.925 },
    routes: [
      { id: 'south-col', name: 'South Col', difficulty: 'Normal', historicalSuccessRate: 0.31 },
      { id: 'north-ridge', name: 'North Ridge', difficulty: 'Normal', historicalSuccessRate: 0.28 },
      { id: 'west-ridge', name: 'West Ridge', difficulty: 'Technical', historicalSuccessRate: 0.15 },
    ],
    imageFolder: 'everest',
  },
  {
    id: 'k2',
    name: 'K2',
    heightM: 8611,
    country: 'Pakistan/China',
    range: 'Karakoram',
    coordinates: { lat: 35.8799, lng: 76.5153 },
    routes: [
      { id: 'abruzzi-spur', name: 'Abruzzi Spur', difficulty: 'Technical', historicalSuccessRate: 0.21 },
      { id: 'cesen-route', name: 'Česen Route', difficulty: 'Technical', historicalSuccessRate: 0.18 },
    ],
    imageFolder: 'k2',
  },
  {
    id: 'kangchenjunga',
    name: 'Kangchenjunga',
    heightM: 8586,
    country: 'Nepal/India',
    range: 'Kangchenjunga Himal',
    coordinates: { lat: 27.7025, lng: 88.1475 },
    routes: [
      { id: 'southwest-face', name: 'Southwest Face', difficulty: 'Normal', historicalSuccessRate: 0.22 },
      { id: 'north-face', name: 'North Face', difficulty: 'Technical', historicalSuccessRate: 0.16 },
    ],
    imageFolder: 'kangchenjunga',
  },
  {
    id: 'lhotse',
    name: 'Lhotse',
    heightM: 8516,
    country: 'Nepal/China',
    range: 'Mahalangur Himal',
    coordinates: { lat: 27.9617, lng: 86.9336 },
    routes: [
      { id: 'west-face', name: 'West Face', difficulty: 'Normal', historicalSuccessRate: 0.26 },
      { id: 'south-face', name: 'South Face', difficulty: 'Extreme', historicalSuccessRate: 0.08 },
    ],
    imageFolder: 'lhotse',
  },
  {
    id: 'makalu',
    name: 'Makalu',
    heightM: 8485,
    country: 'Nepal/China',
    range: 'Mahalangur Himal',
    coordinates: { lat: 27.8897, lng: 87.0886 },
    routes: [
      { id: 'northwest-ridge', name: 'Northwest Ridge', difficulty: 'Normal', historicalSuccessRate: 0.23 },
      { id: 'west-pillar', name: 'West Pillar', difficulty: 'Technical', historicalSuccessRate: 0.14 },
    ],
    imageFolder: 'makalu',
  },
  {
    id: 'cho-oyu',
    name: 'Cho Oyu',
    heightM: 8188,
    country: 'Nepal/China',
    range: 'Mahalangur Himal',
    coordinates: { lat: 28.0942, lng: 86.6608 },
    routes: [
      { id: 'northwest-ridge', name: 'Northwest Ridge', difficulty: 'Normal', historicalSuccessRate: 0.42 },
    ],
    imageFolder: 'cho-oyu',
  },
  {
    id: 'dhaulagiri',
    name: 'Dhaulagiri I',
    heightM: 8167,
    country: 'Nepal',
    range: 'Dhaulagiri Himal',
    coordinates: { lat: 28.6983, lng: 83.4875 },
    routes: [
      { id: 'northeast-ridge', name: 'Northeast Ridge', difficulty: 'Normal', historicalSuccessRate: 0.24 },
      { id: 'south-face', name: 'South Face', difficulty: 'Extreme', historicalSuccessRate: 0.09 },
    ],
    imageFolder: 'dhaulagiri',
  },
  {
    id: 'manaslu',
    name: 'Manaslu',
    heightM: 8163,
    country: 'Nepal',
    range: 'Mansiri Himal',
    coordinates: { lat: 28.5497, lng: 84.5597 },
    routes: [
      { id: 'northeast-face', name: 'Northeast Face', difficulty: 'Normal', historicalSuccessRate: 0.35 },
    ],
    imageFolder: 'manaslu',
  },
  {
    id: 'nanga-parbat',
    name: 'Nanga Parbat',
    heightM: 8126,
    country: 'Pakistan',
    range: 'Nanga Parbat Himal',
    coordinates: { lat: 35.2375, lng: 74.5892 },
    routes: [
      { id: 'rupal-face', name: 'Rupal Face', difficulty: 'Extreme', historicalSuccessRate: 0.12 },
      { id: 'diamir-face', name: 'Diamir Face', difficulty: 'Technical', historicalSuccessRate: 0.19 },
    ],
    imageFolder: 'nanga-parbat',
  },
  {
    id: 'annapurna',
    name: 'Annapurna I',
    heightM: 8091,
    country: 'Nepal',
    range: 'Annapurna Himal',
    coordinates: { lat: 28.5961, lng: 83.8203 },
    routes: [
      { id: 'north-face', name: 'North Face', difficulty: 'Technical', historicalSuccessRate: 0.18 },
      { id: 'south-face', name: 'South Face', difficulty: 'Extreme', historicalSuccessRate: 0.07 },
    ],
    imageFolder: 'annapurna',
  },
  {
    id: 'gasherbrum-i',
    name: 'Gasherbrum I',
    heightM: 8080,
    country: 'Pakistan/China',
    range: 'Karakoram',
    coordinates: { lat: 35.7244, lng: 76.6962 },
    routes: [
      { id: 'japanese-couloir', name: 'Japanese Couloir', difficulty: 'Technical', historicalSuccessRate: 0.2 },
    ],
    imageFolder: 'gasherbrum-i',
  },
  {
    id: 'broad-peak',
    name: 'Broad Peak',
    heightM: 8051,
    country: 'Pakistan/China',
    range: 'Karakoram',
    coordinates: { lat: 35.8122, lng: 76.5653 },
    routes: [
      { id: 'west-spur', name: 'West Spur', difficulty: 'Normal', historicalSuccessRate: 0.28 },
    ],
    imageFolder: 'broad-peak',
  },
  {
    id: 'gasherbrum-ii',
    name: 'Gasherbrum II',
    heightM: 8035,
    country: 'Pakistan/China',
    range: 'Karakoram',
    coordinates: { lat: 35.7572, lng: 76.6533 },
    routes: [
      { id: 'southwest-ridge', name: 'Southwest Ridge', difficulty: 'Normal', historicalSuccessRate: 0.34 },
    ],
    imageFolder: 'gasherbrum-ii',
  },
  {
    id: 'shishapangma',
    name: 'Shishapangma',
    heightM: 8027,
    country: 'China',
    range: 'Jugal Himal',
    coordinates: { lat: 28.3525, lng: 85.7797 },
    routes: [
      { id: 'north-ridge', name: 'North Ridge', difficulty: 'Normal', historicalSuccessRate: 0.38 },
      { id: 'southwest-face', name: 'Southwest Face', difficulty: 'Technical', historicalSuccessRate: 0.17 },
    ],
    imageFolder: 'shishapangma',
  },
]

export function getPeakById(id: string): Peak | undefined {
  return PEAKS.find((peak) => peak.id === id)
}

export function searchPeaks(query: string): Peak[] {
  const normalizedQuery = query.toLowerCase().trim()
  if (!normalizedQuery) return []

  return PEAKS.filter(
    (peak) =>
      peak.name.toLowerCase().includes(normalizedQuery) ||
      peak.country.toLowerCase().includes(normalizedQuery) ||
      peak.range.toLowerCase().includes(normalizedQuery)
  )
}
