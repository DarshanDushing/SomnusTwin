// src/components/RiskGauge.jsx
// Semi-circular gauge showing current risk score

import { motion } from 'framer-motion'
import { riskColor, riskLabel } from '../lib/api'

export default function RiskGauge({ score = 0, size = 160 }) {
  const r = (size / 2) * 0.78
  const cx = size / 2
  const cy = size / 2 + size * 0.1
  const startAngle = -200
  const endAngle = 20
  const totalArc = endAngle - startAngle
  const fillArc = totalArc * score
  const circumference = 2 * Math.PI * r

  function polarToXY(cx, cy, r, angleDeg) {
    const rad = (angleDeg * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  function describeArc(cx, cy, r, startDeg, endDeg) {
    const s = polarToXY(cx, cy, r, startDeg)
    const e = polarToXY(cx, cy, r, endDeg)
    const large = endDeg - startDeg > 180 ? 1 : 0
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
  }

  const trackPath = describeArc(cx, cy, r, startAngle, endAngle)
  const fillPath = describeArc(cx, cy, r, startAngle, startAngle + fillArc)
  const color = riskColor(score)
  
  const badgeClass = score >= 0.65 ? 'severity-severe' : score >= 0.35 ? 'severity-moderate' : 'severity-mild'

  return (
    <div style={{ position: 'relative', width: size, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size * 0.72}`}>
        {/* Track */}
        <path d={trackPath} fill="none" stroke="var(--border)" strokeWidth="8" strokeLinecap="round" />

        {/* Fill */}
        <motion.path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          style={{ filter: `drop-shadow(0 0 6px ${color}80)` }}
        />

        {/* Score text */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize={size * 0.2}
          fontFamily="Space Grotesk, sans-serif"
          fontWeight="700"
          style={{ fontVariantNumeric: 'tabular-nums' }}
        >
          {Math.round(score * 100)}%
        </text>

        {/* Tick labels */}
        {[0, 0.5, 1].map((pct) => {
          const ang = startAngle + totalArc * pct
          const { x, y } = polarToXY(cx, cy, r + 14, ang)
          const labels = ['0', '50', '100']
          return (
            <text
              key={pct}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="var(--text-secondary)"
              fontSize={size * 0.065}
              fontFamily="Inter"
            >
              {labels[pct === 0 ? 0 : pct === 0.5 ? 1 : 2]}
            </text>
          )
        })}
      </svg>
      
      <div 
        className={`chip ${badgeClass}`} 
        style={{ 
          marginTop: -6, 
          fontSize: size * 0.085, 
          padding: '2px 8px',
          fontWeight: 600,
          letterSpacing: '0.02em'
        }}
      >
        {riskLabel(score)} Risk
      </div>
    </div>
  )
}
