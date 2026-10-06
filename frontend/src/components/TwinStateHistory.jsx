// src/components/TwinStateHistory.jsx
// Table/timeline of the persisted twin-state vector across the night.
// Visually distinct from the prediction dashboard.

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { Database, Clock } from 'lucide-react'
import { api, riskColor } from '../lib/api'

function RiskDot({ value }) {
  const color = riskColor(value ?? 0)
  return (
    <span style={{
      display: 'inline-block',
      width: 8, height: 8, borderRadius: '50%',
      background: color,
      boxShadow: `0 0 6px ${color}80`,
      marginRight: 6,
    }} />
  )
}

export default function TwinStateHistory({ subjectId }) {
  const [trajectory, setTrajectory] = useState([])
  const [loading, setLoading] = useState(false)
  const [view, setView] = useState('chart')  // 'chart' | 'table'

  useEffect(() => {
    if (!subjectId) return
    setLoading(true)
    api.getTwinState(subjectId)
      .then(({ data }) => setTrajectory(data.trajectory ?? []))
      .catch(() => setTrajectory([]))
      .finally(() => setLoading(false))
  }, [subjectId])

  if (!subjectId) {
    return (
      <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-secondary)' }}>
        <Database size={28} style={{ margin: '0 auto 8px', opacity: 0.4 }} />
        <p style={{ fontSize: '0.85rem' }}>Select a subject to view twin state history</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', padding: '20px 0' }}>
        <div style={{
          width: 16, height: 16, borderRadius: '50%',
          border: '2px solid var(--primary)', borderTopColor: 'transparent',
          animation: 'spin 0.8s linear infinite',
        }} />
        Loading twin state trajectory from SQLite...
      </div>
    )
  }

  if (!trajectory.length) {
    return (
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', padding: '16px 0' }}>
        No persisted twin states found. Populate by running the training pipeline and API.
      </p>
    )
  }

  const chartData = trajectory.map(t => ({
    minute: t.minute,
    risk: t.risk_score ? parseFloat((t.risk_score * 100).toFixed(1)) : null,
    hr: t.hr_current ? parseFloat(t.hr_current.toFixed(1)) : null,
    spo2: t.spo2_current ? parseFloat(t.spo2_current.toFixed(1)) : null,
    apnea: t.apnea_now ? 100 : 0,
    gru_norm: t.gru_hidden_norm ? parseFloat(t.gru_hidden_norm.toFixed(2)) : null,
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Database size={16} style={{ color: 'var(--accent-lavender)' }} />
          <h3 style={{ fontFamily: 'Space Grotesk', fontSize: '0.95rem' }}>
            Twin State Trajectory
          </h3>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginLeft: 4 }}>
            {trajectory.length} timesteps · SQLite-persisted
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className={`tab ${view === 'chart' ? 'active' : ''}`}
            style={{ padding: '5px 14px', fontSize: '0.75rem' }}
            onClick={() => setView('chart')}
          >
            Chart
          </button>
          <button
            className={`tab ${view === 'table' ? 'active' : ''}`}
            style={{ padding: '5px 14px', fontSize: '0.75rem' }}
            onClick={() => setView('table')}
          >
            Table
          </button>
        </div>
      </div>

      {view === 'chart' ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card"
          style={{ padding: 16 }}
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="minute" tick={{ fontSize: 9 }} label={{ value: 'Minute', position: 'insideBottomRight', fontSize: 9, fill: 'var(--text-secondary)' }} />
              <YAxis yAxisId="risk" domain={[0, 100]} tick={{ fontSize: 9 }} />
              <YAxis yAxisId="hr" orientation="right" domain={[40, 130]} tick={{ fontSize: 9 }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-alt)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: '0.72rem',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.72rem' }} />
              <ReferenceLine yAxisId="risk" y={55} stroke="var(--accent-amber)" strokeDasharray="4 2" />
              <Line yAxisId="risk" type="monotone" dataKey="risk" name="Risk %" stroke="var(--accent-coral)" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line yAxisId="risk" type="monotone" dataKey="apnea" name="Apnea label" stroke="var(--accent-coral)" strokeWidth={1} dot={false} strokeDasharray="4 2" opacity={0.5} isAnimationActive={false} />
              <Line yAxisId="hr" type="monotone" dataKey="hr" name="HR" stroke="var(--accent-lavender)" strokeWidth={1} dot={false} isAnimationActive={false} />
              <Line yAxisId="hr" type="monotone" dataKey="spo2" name="SpO₂" stroke="var(--primary)" strokeWidth={1} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: 8, textAlign: 'center' }}>
            Risk scores are the model's real-time predictions, stored per-minute in SQLite.
            Dashed line = clinical annotation (apnea minute). Dashed amber = 55% risk threshold.
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 360 }}
        >
          <table style={{
            width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem',
            fontFamily: 'Inter, monospace',
          }}>
            <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
              <tr style={{ background: 'var(--surface-alt)' }}>
                {['Min', 'Risk%', 'HR', 'HRV', 'SpO₂', 'GRU‖h‖', 'Apnea', 'Saved at'].map(h => (
                  <th key={h} style={{
                    padding: '8px 10px', textAlign: 'left',
                    color: 'var(--text-secondary)', fontWeight: 500,
                    borderBottom: '1px solid var(--border)',
                    whiteSpace: 'nowrap',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trajectory.map((t, idx) => (
                <motion.tr
                  key={t.minute}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: Math.min(idx * 0.01, 0.3) }}
                  style={{
                    background: t.apnea_now ? 'rgba(242,112,92,0.06)' : idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                    borderBottom: '1px solid rgba(38,51,79,0.5)',
                  }}
                >
                  <td style={{ padding: '6px 10px', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {t.minute}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <RiskDot value={t.risk_score} />
                    <span style={{ color: riskColor(t.risk_score ?? 0), fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
                      {t.risk_score != null ? `${(t.risk_score * 100).toFixed(1)}%` : '—'}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums' }}>
                    {t.hr_current?.toFixed(0) ?? '—'}
                  </td>
                  <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums' }}>
                    {t.hrv_current?.toFixed(1) ?? '—'}
                  </td>
                  <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums' }}>
                    {t.spo2_current?.toFixed(1) ?? '—'}
                  </td>
                  <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums', color: 'var(--accent-lavender)' }}>
                    {t.gru_hidden_norm?.toFixed(2) ?? '—'}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    {t.apnea_now ? (
                      <span style={{ color: 'var(--accent-coral)', fontSize: '0.7rem' }}>⚠ Yes</span>
                    ) : (
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.7rem' }}>—</span>
                    )}
                  </td>
                  <td style={{ padding: '6px 10px', color: 'var(--text-secondary)', fontSize: '0.68rem' }}>
                    {t.created_at ? t.created_at.slice(0, 16).replace('T', ' ') : '—'}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}
    </div>
  )
}
