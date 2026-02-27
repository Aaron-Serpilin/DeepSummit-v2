# DeepSummit — Frontend Architecture

This document defines the complete frontend architecture, design system, user flows, and implementation guidelines for the DeepSummit v2 web application.

---

## Design Vision

**"Expedition Mission Control meets Luxury Data Viz"** — a dark, atmospheric interface combining technical precision with refined elegance. Think NASA console aesthetics crossed with high-end mountaineering gear design.

The frontend achieves:
- **Immersive map experience**: Dark 2D world map with glowing peak markers
- **FIFA-style peak reveals**: Cinematic mountain image displays with parallax depth
- **Progressive disclosure**: Step-by-step prediction wizard
- **Data-rich results**: Clear probability gauges and SHAP feature attribution

---

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Framework | React 18+ | Component-based UI |
| Build Tool | Vite | Fast development and optimized builds |
| Language | TypeScript | Type safety across the codebase |
| Styling | Tailwind CSS | Utility-first styling |
| Components | shadcn/ui | Accessible, customizable base components |
| Map | Mapbox GL JS (react-map-gl) | Dark world map with fly-to animations |
| Animation | Framer Motion | Page transitions and micro-interactions |
| Charts | Recharts | SHAP value visualization |
| State | React useState / useReducer | Simple local state (no Redux) |
| Data Fetching | TanStack Query | Server state management when backend exists |
| Hosting | Firebase Hosting | Static asset serving |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DeepSummit Frontend                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   React + Vite + TypeScript                                      │
│   ├── UI: shadcn/ui + Tailwind CSS                               │
│   ├── Map: Mapbox GL JS (react-map-gl)                           │
│   ├── Animations: Framer Motion                                  │
│   ├── Charts: Recharts (SHAP visualization)                      │
│   ├── State: React useState/useReducer (no Redux needed)         │
│   ├── Data Fetching: TanStack Query (when backend exists)        │
│   └── Hosting: Firebase Hosting                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure (Atomic Design)

```
frontend/
├── app/
│   ├── components/
│   │   ├── atoms/                    # Smallest reusable components
│   │   │   ├── Button/
│   │   │   │   ├── Button.tsx
│   │   │   │   └── index.ts
│   │   │   ├── Badge/
│   │   │   ├── Input/
│   │   │   ├── GlowDot/              # Peak markers on map
│   │   │   ├── ProgressRing/         # SVG circular progress
│   │   │   └── Tooltip/
│   │   │
│   │   ├── molecules/                # Composed components
│   │   │   ├── Map/
│   │   │   │   ├── Map.tsx           # Mapbox integration
│   │   │   │   ├── PeakMarker.tsx    # Individual peak marker
│   │   │   │   └── index.ts
│   │   │   ├── MountainDisplay/
│   │   │   │   ├── MountainDisplay.tsx
│   │   │   │   ├── ParallaxLayer.tsx
│   │   │   │   └── index.ts
│   │   │   ├── WeatherCard/          # Wind/temp/pressure cards
│   │   │   ├── ProbabilityGauge/     # Circular progress gauge
│   │   │   ├── SHAPChart/            # Horizontal bar chart
│   │   │   ├── WizardStep/           # Form step container
│   │   │   ├── PeakSearch/           # Search input + typeahead
│   │   │   └── Header/               # App header with logo
│   │   │
│   │   └── pages/                    # Full page layouts
│   │       └── Home/
│   │           ├── Home.tsx          # Main single-page app
│   │           ├── MapView.tsx       # Map state view
│   │           ├── PeakView.tsx      # Peak + wizard view
│   │           ├── ResultsView.tsx   # Prediction results view
│   │           └── index.ts
│   │
│   ├── data/
│   │   └── peaks.ts                  # Static peak information (14 peaks)
│   │
│   ├── assets/
│   │   ├── mountains/                # AI-generated mountain images
│   │   │   ├── everest/
│   │   │   │   ├── layer-sky.png
│   │   │   │   ├── layer-mountain.png
│   │   │   │   └── layer-foreground.png
│   │   │   ├── k2/
│   │   │   └── ...                   # 14 peaks total
│   │   └── icons/
│   │
│   ├── hooks/
│   │   ├── useParallax.ts            # Mouse parallax effect
│   │   ├── useMapFlyTo.ts            # Map animation control
│   │   └── useWizardState.ts         # Multi-step form state
│   │
│   ├── types/
│   │   ├── peak.ts                   # Peak type definitions
│   │   ├── prediction.ts             # Prediction request/response types
│   │   └── wizard.ts                 # Wizard step types
│   │
│   ├── lib/
│   │   └── utils.ts                  # shadcn/ui utility (cn function)
│   │
│   ├── styles/
│   │   └── globals.css               # Tailwind directives + custom CSS
│   │
│   ├── App.tsx                       # Root component
│   └── main.tsx                      # Vite entry point
│
├── public/
│   ├── favicon.ico
│   └── deepsummit-v2.png
│
├── package.json
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── .env.local                        # Mapbox token, etc.
```

### Atomic Design Hierarchy

| Level | Description | Examples |
|-------|-------------|----------|
| **Atoms** | Smallest, single-purpose UI elements | Button, Badge, Input, GlowDot |
| **Molecules** | Components composed of atoms | Map, MountainDisplay, SHAPChart |
| **Pages** | Full page layouts that orchestrate molecules | Home (contains MapView, PeakView, ResultsView) |

---

## User Flow

The application is a single-page experience with three main views connected by smooth Framer Motion transitions.

```
╔══════════════════════════════════════════════════════════════════╗
║  VIEW 1: MAP VIEW (Landing)                                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  "DeepSummit" logo                    [Search peak...]   │   ║
║  ├──────────────────────────────────────────────────────────┤   ║
║  │                                                          │   ║
║  │                    DARK WORLD MAP                        │   ║
║  │                    (Mapbox dark style)                   │   ║
║  │                                                          │   ║
║  │         ● Everest                                        │   ║
║  │            ● K2                                          │   ║
║  │         ● Kangchenjunga     (14 glowing cyan dots)       │   ║
║  │         ...                                              │   ║
║  │                                                          │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  User clicks a dot OR types peak name → fly-to animation        ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              │ (Framer Motion fade/zoom transition)
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  VIEW 2: PEAK VIEW (FIFA-style layout)                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  ← Back to Map              "Mount Everest"    8,849m    │   ║
║  ├─────────────────────────┬────────────────────────────────┤   ║
║  │                         │                                │   ║
║  │   ┌─────────────────┐   │   ┌────────────────────────┐   │   ║
║  │   │                 │   │   │  PREDICTION WIZARD     │   │   ║
║  │   │   MOUNTAIN      │   │   │                        │   │   ║
║  │   │   IMAGE         │   │   │  Step 1: Route         │   │   ║
║  │   │   (parallax)    │   │   │  [ ] South Col         │   │   ║
║  │   │                 │   │   │  [ ] North Ridge       │   │   ║
║  │   │                 │   │   │                        │   │   ║
║  │   └─────────────────┘   │   │  [Next →]              │   │   ║
║  │                         │   └────────────────────────┘   │   ║
║  │   Wind: 45 km/h         │                                │   ║
║  │   Temp: -22°C           │                                │   ║
║  │   Pressure: 612 hPa     │                                │   ║
║  │                         │                                │   ║
║  └─────────────────────────┴────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              │ (Complete wizard steps 1-3)
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  VIEW 3: RESULTS                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │       ┌─────────────┐   │   ┌────────────────────────┐   │   ║
║  │       │   73%       │   │   │  SHAP ANALYSIS         │   │   ║
║  │       │  ──────     │   │   │  ▓▓▓▓▓░░ Wind          │   │   ║
║  │       │  MODERATE   │   │   │  ▓▓▓▓░░░ Experience    │   │   ║
║  │       │    RISK     │   │   │  ▓▓▓░░░░ Oxygen        │   │   ║
║  │       └─────────────┘   │   │  ▓▓░░░░░ Team Size     │   │   ║
║  │                         │   └────────────────────────┘   │   ║
║  │   Weather Summary       │                                │   ║
║  │   ─────────────────     │   [← New Prediction]           │   ║
║  │   Wind: 48 km/h         │   [Share Results]              │   ║
║  │   Temp: -22°C           │                                │   ║
║  └─────────────────────────┴────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════╝
```

### State Machine

```
       ┌──────────────────┐
       │                  │
       │    MAP VIEW      │
       │                  │
       └────────┬─────────┘
                │
                │ selectPeak(peakId) or searchPeak(name)
                │ → triggers flyTo animation
                ▼
       ┌──────────────────┐
       │                  │
       │    PEAK VIEW     │ ◄───── goBack()
       │    (wizard)      │
       │                  │
       └────────┬─────────┘
                │
                │ submitPrediction()
                ▼
       ┌──────────────────┐
       │                  │
       │   RESULTS VIEW   │ ◄───── newPrediction() → PEAK VIEW
       │                  │
       └──────────────────┘
```

---

## Design System

### Color Palette

```
┌─────────────────────────────────────────────────────────────────┐
│                        COLOR PALETTE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   BACKGROUNDS                                                    │
│   ├── Primary:     #0A0A0F   (near black, main background)       │
│   ├── Secondary:   #141420   (cards, elevated surfaces)          │
│   └── Tertiary:    #1E1E2E   (hover states, borders)             │
│                                                                  │
│   ACCENT COLORS                                                  │
│   ├── Cyan:        #00D4FF   (primary accent, glows, highlights) │
│   ├── Cyan Dim:    #0099B8   (secondary cyan, inactive states)   │
│   └── Orange:      #FF6B35   (warnings, "High Risk" badges)      │
│                                                                  │
│   TEXT                                                           │
│   ├── Primary:     #FFFFFF   (headings)                          │
│   ├── Secondary:   #A0A0B0   (body text, labels)                 │
│   └── Muted:       #606070   (placeholders, disabled)            │
│                                                                  │
│   RISK BADGES                                                    │
│   ├── Low:         #22C55E   (green)                             │
│   ├── Moderate:    #F59E0B   (amber/orange)                      │
│   ├── High:        #EF4444   (red)                               │
│   └── Extreme:     #DC2626   (dark red)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tailwind Configuration

```javascript
// tailwind.config.ts
export default {
  theme: {
    extend: {
      colors: {
        // Backgrounds
        'ds-bg': '#0A0A0F',
        'ds-bg-secondary': '#141420',
        'ds-bg-tertiary': '#1E1E2E',

        // Accents
        'ds-cyan': '#00D4FF',
        'ds-cyan-dim': '#0099B8',
        'ds-orange': '#FF6B35',

        // Text
        'ds-text': '#FFFFFF',
        'ds-text-secondary': '#A0A0B0',
        'ds-text-muted': '#606070',

        // Risk levels
        'ds-risk-low': '#22C55E',
        'ds-risk-moderate': '#F59E0B',
        'ds-risk-high': '#EF4444',
        'ds-risk-extreme': '#DC2626',
      },
    },
  },
}
```

### Typography

| Element | Font | Size | Weight | Use Case |
|---------|------|------|--------|----------|
| Display | Space Grotesk | 48px | 700 | "DeepSummit" title |
| H1 | Space Grotesk | 32px | 600 | Peak name |
| H2 | Space Grotesk | 24px | 600 | Section headers |
| Body | Inter | 16px | 400 | Default text |
| Small | Inter | 14px | 400 | Labels, captions |
| Data | JetBrains Mono | 16px | 500 | Numbers, percentages, altitude |

**Font Loading** (Google Fonts):
```html
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**Why Space Grotesk?** The frontend-design skill emphasizes avoiding generic fonts (Inter, Roboto, Arial). Space Grotesk has a technical, geometric quality that aligns with the "mission control" aesthetic while remaining highly readable.

---

## Component Specifications

### Map (molecule)

**Purpose**: Dark world map showing all 14 eight-thousanders as interactive markers.

**Technology**: Mapbox GL JS via `react-map-gl`

**Configuration**:
```typescript
const mapStyle = 'mapbox://styles/mapbox/dark-v11';

const initialViewState = {
  longitude: 86.9250,  // Center on Himalayas
  latitude: 27.9881,
  zoom: 4,
  pitch: 0,
  bearing: 0,
};
```

**Interactions**:
- Click marker → `flyTo()` with smooth animation (duration: 2000ms)
- Search input → triggers same `flyTo()` behavior
- Hover marker → show peak name tooltip

**Marker Style**:
```css
/* Glowing cyan dot */
.peak-marker {
  width: 12px;
  height: 12px;
  background: #00D4FF;
  border-radius: 50%;
  box-shadow: 0 0 20px #00D4FF, 0 0 40px #00D4FF80;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.8; }
}
```

---

### MountainDisplay (molecule)

**Purpose**: FIFA-style mountain image with parallax depth effect.

**Approach**: Layered PNGs with CSS transforms.

**Layer Structure** (per peak):
```
assets/mountains/everest/
├── layer-sky.png          # Background (moves least)
├── layer-mountain.png     # Middle layer (main peak)
└── layer-foreground.png   # Foreground elements (moves most)
```

**Parallax Implementation**:
```typescript
// useParallax.ts
export function useParallax(ref: RefObject<HTMLDivElement>) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const x = (e.clientX - rect.left - rect.width / 2) / rect.width;
      const y = (e.clientY - rect.top - rect.height / 2) / rect.height;
      setOffset({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [ref]);

  return offset;
}
```

**Layer Movement**:
```typescript
// Depth multipliers: higher = more movement
const layers = [
  { src: 'layer-sky.png', depth: 0.02 },        // Subtle
  { src: 'layer-mountain.png', depth: 0.05 },   // Medium
  { src: 'layer-foreground.png', depth: 0.1 },  // Most movement
];

// Transform calculation
const transform = `translate(${offset.x * depth * 100}px, ${offset.y * depth * 100}px)`;
```

**Entrance Animation** (Framer Motion):
```typescript
<motion.div
  initial={{ scale: 0.9, opacity: 0 }}
  animate={{ scale: 1, opacity: 1 }}
  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
>
  {/* Mountain layers */}
</motion.div>
```

---

### ProbabilityGauge (molecule)

**Purpose**: Circular gauge showing success probability (0-100%).

**Implementation**: SVG `<circle>` with `stroke-dasharray` animation.

```typescript
interface ProbabilityGaugeProps {
  value: number;          // 0-100
  riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME';
}

const circumference = 2 * Math.PI * radius;
const dashOffset = circumference * (1 - value / 100);
```

**Visual Design**:
- Thick stroke (8-10px)
- Gradient fill from cyan to color matching risk level
- Center displays: percentage (large, JetBrains Mono) + risk badge (below)
- Animated fill on mount (0% → actual value over 1.2s)

---

### SHAPChart (molecule)

**Purpose**: Horizontal bar chart showing feature contributions to the prediction.

**Implementation**: Recharts `<BarChart>` with horizontal layout.

**Data Structure**:
```typescript
interface SHAPValue {
  feature: string;
  value: number;      // Negative or positive
  contribution: 'positive' | 'negative';
}
```

**Visual Design**:
- Negative bars extend left (red tint) → feature decreases success probability
- Positive bars extend right (green tint) → feature increases success probability
- Sorted by absolute magnitude (most impactful at top)
- Labels show feature name + impact value

---

### WizardStep (molecule)

**Purpose**: Container for each step in the prediction wizard.

**Steps**:
1. **Route Selection**: Radio buttons for available routes (Normal, variations)
2. **Date & Team**: Date picker, team size, commercial/alpine toggle
3. **Climber Profile**: Age input, gender, oxygen planned, prior experience

**Step Transitions** (Framer Motion):
```typescript
<AnimatePresence mode="wait">
  <motion.div
    key={currentStep}
    initial={{ x: 50, opacity: 0 }}
    animate={{ x: 0, opacity: 1 }}
    exit={{ x: -50, opacity: 0 }}
    transition={{ duration: 0.3 }}
  >
    {/* Step content */}
  </motion.div>
</AnimatePresence>
```

**Progress Indicator**: Horizontal dots showing current position (Step 1 of 3).

---

### WeatherCard (molecule)

**Purpose**: Display current weather conditions at the peak.

**Metrics**:
- Wind speed (km/h)
- Temperature (°C)
- Atmospheric pressure (hPa)
- (Optional) Precipitation, visibility

**Visual Style**: Dark card with icon + value + unit. Use color coding for severity (e.g., extreme wind turns orange/red).

---

## Static Data

### Peak Information (14 Eight-Thousanders)

```typescript
// data/peaks.ts
export interface Peak {
  id: string;
  name: string;
  height_m: number;
  country: string;
  range: string;
  coordinates: { lat: number; lng: number };
  routes: Route[];
  image_folder: string;
}

export interface Route {
  id: string;
  name: string;
  difficulty: 'Normal' | 'Technical' | 'Extreme';
  historical_success_rate?: number;
}

export const PEAKS: Peak[] = [
  {
    id: 'everest',
    name: 'Mount Everest',
    height_m: 8849,
    country: 'Nepal/China',
    range: 'Mahalangur Himal',
    coordinates: { lat: 27.9881, lng: 86.9250 },
    routes: [
      { id: 'south-col', name: 'South Col', difficulty: 'Normal' },
      { id: 'north-ridge', name: 'North Ridge', difficulty: 'Normal' },
    ],
    image_folder: 'everest',
  },
  // ... remaining 13 peaks
];
```

### All 14 Peaks Reference

| Peak | Height | Country | Coordinates |
|------|--------|---------|-------------|
| Everest | 8,849m | Nepal/China | 27.9881°N, 86.9250°E |
| K2 | 8,611m | Pakistan/China | 35.8799°N, 76.5153°E |
| Kangchenjunga | 8,586m | Nepal/India | 27.7025°N, 88.1475°E |
| Lhotse | 8,516m | Nepal/China | 27.9617°N, 86.9336°E |
| Makalu | 8,485m | Nepal/China | 27.8897°N, 87.0886°E |
| Cho Oyu | 8,188m | Nepal/China | 28.0942°N, 86.6608°E |
| Dhaulagiri I | 8,167m | Nepal | 28.6983°N, 83.4875°E |
| Manaslu | 8,163m | Nepal | 28.5497°N, 84.5597°E |
| Nanga Parbat | 8,126m | Pakistan | 35.2375°N, 74.5892°E |
| Annapurna I | 8,091m | Nepal | 28.5961°N, 83.8203°E |
| Gasherbrum I | 8,080m | Pakistan/China | 35.7244°N, 76.6962°E |
| Broad Peak | 8,051m | Pakistan/China | 35.8122°N, 76.5653°E |
| Gasherbrum II | 8,035m | Pakistan/China | 35.7572°N, 76.6533°E |
| Shishapangma | 8,027m | China | 28.3525°N, 85.7797°E |

---

## Key Technical Decisions

### Mapbox GL JS over Leaflet or CesiumJS

**Decision**: Use Mapbox GL JS with the `dark-v11` style.

**Rationale**:
- Built-in dark map styles eliminate custom styling work
- `flyTo()` animations are smooth and configurable out of the box
- WebGL rendering is performant for pan/zoom
- Free tier (50k map loads/month) is generous for a portfolio project
- CesiumJS terrain was evaluated but rejected — satellite textures at high zoom aren't photorealistic enough for the cinematic vision; a stylized 2D map + AI-generated mountain images achieves a cleaner aesthetic

### AI-Generated Mountain Images over Real-Time 3D

**Decision**: Use pre-generated, stylized mountain images (Midjourney/DALL-E 3) instead of CesiumJS terrain or Three.js 3D models.

**Rationale**:
- CesiumJS terrain produces recognizable but not beautiful peaks (Google Earth quality, not cinematic)
- Three.js 3D models would require sourcing/creating 14 high-quality mountain meshes
- AI image generation allows for a consistent artistic style across all 14 peaks
- Layered PNGs for parallax are performant and achievable without complex 3D pipeline
- 14 images is a manageable asset set for MVP

### Single-Page with View States over Multi-Page Routing

**Decision**: One continuous page that transforms between three view states (Map → Peak → Results).

**Rationale**:
- Fluid, app-like experience with Framer Motion transitions
- No URL routing complexity for MVP
- State can be managed with simple `useState` (no router state synchronization)
- Easy to add URL routing later if needed (e.g., shareable peak links)

### Space Grotesk over Generic Sans-Serif

**Decision**: Space Grotesk for headings and display text.

**Rationale**:
- The frontend-design skill explicitly recommends avoiding Inter, Roboto, Arial
- Space Grotesk has a technical, geometric quality matching "mission control" aesthetic
- Pairs well with JetBrains Mono for data/numbers
- Free on Google Fonts

---

## Animation Guidelines

### Page Transitions (Framer Motion)

```typescript
// Shared transition config for consistency
export const pageTransition = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
};
```

### Map Fly-To

```typescript
mapRef.current?.flyTo({
  center: [peak.coordinates.lng, peak.coordinates.lat],
  zoom: 8,
  pitch: 45,
  bearing: 0,
  duration: 2000,
  essential: true,
});
```

### Micro-Interactions

| Element | Interaction | Animation |
|---------|-------------|-----------|
| Peak marker | Hover | Scale 1.2× + glow increase |
| Button | Click | Scale 0.95× (press) |
| Wizard step | Enter/Exit | Slide X + fade |
| Results gauge | Mount | Draw stroke 0 → 100% |
| SHAP bars | Mount | Stagger reveal (50ms delay each) |

---

## Responsive Considerations

**Primary target**: Desktop (1280px+)

**Breakpoints**:
```css
sm: 640px   /* Mobile landscape */
md: 768px   /* Tablet */
lg: 1024px  /* Small desktop */
xl: 1280px  /* Standard desktop */
2xl: 1536px /* Large screens */
```

**Layout Adaptation**:
- **Desktop**: Side-by-side layout (mountain image | wizard/results)
- **Tablet/Mobile**: Stacked layout (mountain image above, content below)
- Map view should remain usable at all sizes (markers may need labels on mobile)

---

## References

- [Mapbox GL JS Documentation](https://docs.mapbox.com/mapbox-gl-js/)
- [react-map-gl](https://visgl.github.io/react-map-gl/)
- [Framer Motion](https://www.framer.com/motion/)
- [shadcn/ui](https://ui.shadcn.com/)
- [Recharts](https://recharts.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Space Grotesk Font](https://fonts.google.com/specimen/Space+Grotesk)
