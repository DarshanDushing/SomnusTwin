// src/components/StatCards.jsx
// Current HR/HRV/SpO2 stat cards with sparklines

import { motion } from 'framer-motion'
import { LineChart, Line, ResponsiveContainer } from 'recharts'
import { Heart, Activity, Wind } from 'lucide-react'

function Sparkline({ data, color }) {
  return (
    <ResponsiveContainer width="100%" height={36}>
      <LineChart data={data.map((v, i) => ({ i, v }))}>
        <Line
          type="monotone" dataKey="v"
          stroke={color} strokeWidth={1.5}
          dot={false} isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function StatCard({ icon: Icon, label, value, unit, color, trend, sparkData, delay = 0 }) {
  const trendDir = trend > 0.1 ? '↑' : trend < -0.1 ? '↓' : '→'
  const trendColor = label === 'SpO₂'
    ? (trend < 0 ? 'var(--accent-coral)' : 'var(--primary)')
    : (trend > 0 ? 'var(--accent-amber)' : 'var(--primary)')

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 220 }}
      className="card"
      style={{ padding: '12px 16px', flex: 1, minWidth: 120 }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{
            width: 28, height: 28, borderRadius: '50%',
            background: `${color}18`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={14} style={{ color }} />
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            {label}
          </span>
        </div>
        {trend != null && (
          <span style={{ fontSize: '0.75rem', color: trendColor, fontWeight: 600 }}>
            {trendDir}
          </span>
        )}
      </div>

      <motion.p
        key={value}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        style={{
          fontFamily: 'Space Grotesk, sans-serif',
          fontSize: '1.5rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          fontVariantNumeric: 'tabular-nums',
          lineHeight: 1,
        }}
      >
        {value != null ? value : '—'}
        <span style={{ fontSize: '0.75rem', fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 3 }}>
          {unit}
        </span>
      </motion.p>

      {sparkData?.length > 2 && (
        <div style={{ marginTop: 6, opacity: 0.8 }}>
          <Sparkline data={sparkData} color={color} />
        </div>
      )}
    </motion.div>
  )
}

export default function StatCards({ signal = [], currentMinute = 0 }) {
  const recent = signal.slice(Math.max(0, currentMinute - 14), currentMinute + 1)
  const cur = recent[recent.length - 1] ?? {}
  const prev = recent[recent.length - 4] ?? {}

  const hrTrend = cur.hr != null && prev.hr != null ? cur.hr - prev.hr : null
  const hrvTrend = cur.hrv != null && prev.hrv != null ? cur.hrv - prev.hrv : null
  const spo2Trend = cur.spo2 != null && prev.spo2 != null ? cur.spo2 - prev.spo2 : null

  const hrSpark = recent.filter(r => r.hr != null).map(r => r.hr)
  const hrvSpark = recent.filter(r => r.hrv != null).map(r => r.hrv)
  const spo2Spark = recent.filter(r => r.spo2 != null).map(r => r.spo2)

  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <StatCard
        icon={Heart}
        label="HR"
        value={cur.hr != null ? Math.round(cur.hr) : null}
        unit="bpm"
        color="var(--accent-lavender)"
        trend={hrTrend}
        sparkData={hrSpark}
        delay={0}
      />
      <StatCard
        icon={Activity}
        label="HRV"
        value={cur.hrv != null ? Math.round(cur.hrv) : null}
        unit="ms"
        color="var(--primary)"
        trend={hrvTrend}
        sparkData={hrvSpark}
        delay={0.05}
      />
      <StatCard
        icon={Wind}
        label="SpO₂"
        value={cur.spo2 != null ? cur.spo2.toFixed(1) : null}
        unit="%"
        color={cur.spo2 != null && cur.spo2 < 90 ? 'var(--accent-coral)' : cur.spo2 < 95 ? 'var(--accent-amber)' : 'var(--primary)'}
        trend={spo2Trend}
        sparkData={spo2Spark}
        delay={0.10}
      />
    </div>
  )
}
