// src/components/ReasonCodes.jsx
// SHAP-driven plain-language reason codes with staggered chip animation

import { motion, AnimatePresence } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

function ReasonChip({ reason, index, isAnimated }) {
  const isRisk = reason.shap_value > 0
  const isNeutral = Math.abs(reason.shap_value) < 0.01

  const color = isNeutral
    ? 'var(--text-secondary)'
    : isRisk
    ? 'var(--accent-coral)'
    : 'var(--primary)'

  const bg = isNeutral
    ? 'rgba(138, 153, 179, 0.1)'
    : isRisk
    ? 'rgba(242, 112, 92, 0.1)'
    : 'rgba(111, 227, 208, 0.1)'

  const borderColor = isNeutral
    ? 'rgba(138, 153, 179, 0.2)'
    : isRisk
    ? 'rgba(242, 112, 92, 0.25)'
    : 'rgba(111, 227, 208, 0.25)'

  const Icon = isNeutral ? Minus : isRisk ? TrendingUp : TrendingDown

  return (
    <motion.div
      layout={isAnimated}
      initial={isAnimated ? { opacity: 0, x: -12, scale: 0.95 } : { opacity: 1, x: 0, scale: 1 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={isAnimated ? { opacity: 0, x: 12, scale: 0.95 } : { opacity: 0, transition: { duration: 0 } }}
      transition={isAnimated ? { delay: index * 0.06, type: 'spring', stiffness: 280 } : { duration: 0 }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 12px',
        borderRadius: 10,
        background: bg,
        border: `1px solid ${borderColor}`,
        fontSize: '0.8rem',
      }}
    >
      <div style={{
        flexShrink: 0,
        width: 24,
        height: 24,
        borderRadius: '50%',
        background: isRisk ? 'rgba(242,112,92,0.15)' : 'rgba(111,227,208,0.15)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <Icon size={12} style={{ color }} />
      </div>
      <div style={{ flex: 1 }}>
        <p style={{ color: 'var(--text-primary)', lineHeight: 1.3, fontSize: '0.78rem' }}>
          {reason.label}
        </p>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.68rem', marginTop: 2 }}>
          SHAP: {reason.shap_value > 0 ? '+' : ''}{reason.shap_value.toFixed(3)}
        </p>
      </div>
      <span style={{
        fontSize: '0.68rem',
        fontWeight: 600,
        color,
        background: bg,
        padding: '2px 6px',
        borderRadius: 6,
        border: `1px solid ${borderColor}`,
        flexShrink: 0,
      }}>
        {reason.direction}
      </span>
    </motion.div>
  )
}

export default function ReasonCodes({ reasons = [], loading = false, isAnimated = true }) {
  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{
            height: 48, borderRadius: 10,
            background: 'var(--surface-alt)',
            animation: 'pulse 1.5s ease infinite',
            opacity: 0.5,
          }} />
        ))}
      </div>
    )
  }

  if (!reasons.length) {
    return (
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textAlign: 'center', padding: '16px 0' }}>
        Select a subject and start playback to see SHAP-driven explanations
      </p>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <AnimatePresence mode="popLayout">
        {reasons.map((r, i) => (
          <ReasonChip key={r.feature} reason={r} index={i} isAnimated={isAnimated} />
        ))}
      </AnimatePresence>
      <p style={{ fontSize: '0.67rem', color: 'var(--text-secondary)', marginTop: 4 }}>
        SHAP values computed against real LightGBM model (TreeExplainer)
      </p>
    </div>
  )
}
