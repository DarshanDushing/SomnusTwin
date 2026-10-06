// src/components/TwinAvatar.jsx
import { motion } from 'framer-motion'
import { riskColor } from '../lib/api'

export default function TwinAvatar({ riskScore = 0 }) {
  const color = riskColor(riskScore)
  const riskPct = Math.round(riskScore * 100)
  
  const SIZE = 200
  const CX = SIZE / 2
  const CY = SIZE / 2
  const R = 85
  const CIRCUM = 2 * Math.PI * R

  return (
    <div style={{ position: 'relative', width: SIZE, height: SIZE, margin: '0 auto' }}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}>
        {/* Background Track */}
        <circle 
          cx={CX} 
          cy={CY} 
          r={R} 
          fill="none" 
          stroke="var(--border)" 
          strokeWidth="8" 
          opacity="0.3" 
        />
        {/* Foreground Progress Arc */}
        <circle 
          cx={CX} 
          cy={CY} 
          r={R} 
          fill="none" 
          stroke={color} 
          strokeWidth="8"
          strokeDasharray={CIRCUM}
          strokeDashoffset={CIRCUM * (1 - riskScore)}
          strokeLinecap="round"
          transform={`rotate(-90 ${CX} ${CY})`}
          style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.8s ease' }}
        />
      </svg>
      
      {/* Centered Text Overlay */}
      <div style={{
        position: 'absolute', 
        top: '50%', 
        left: '50%', 
        transform: 'translate(-50%, -50%)',
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center',
        width: '100%',
        height: '100%'
      }}>
        <motion.span
          key={riskPct}
          initial={{ scale: 0.7, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300 }}
          style={{
            fontFamily: 'Space Grotesk, sans-serif',
            fontSize: '1.8rem',
            fontWeight: 700,
            color,
            lineHeight: 1,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {riskPct}%
        </motion.span>
        <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
          risk
        </span>
      </div>
    </div>
  )
}
