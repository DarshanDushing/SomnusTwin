// src/components/WhatIfPanel.jsx
// Sliders for HR/HRV/SpO2 → calls /simulate endpoint → shows live risk.
// Proves the model isn't hardcoded by showing clearly different outputs.

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sliders, RefreshCw } from 'lucide-react'
import { api, riskColor, riskLabel } from '../lib/api'
import RiskGauge from './RiskGauge'
import ReasonCodes from './ReasonCodes'

const PRESETS = [
  { label: 'Healthy baseline', hr: 62, hrv: 45, spo2: 98.5, icon: '😴' },
  { label: 'Mild risk', hr: 78, hrv: 22, spo2: 95.5, icon: '😐' },
  { label: 'High risk', hr: 95, hrv: 8, spo2: 87.0, icon: '⚠️' },
]

function SliderRow({ label, id, value, min, max, step, unit, onChange, color }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <label htmlFor={id} style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          {label}
        </label>
        <motion.span
          key={value}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          style={{
            fontSize: '0.9rem',
            fontWeight: 700,
            color,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {typeof value === 'number' ? value.toFixed(step < 1 ? 1 : 0) : value} {unit}
        </motion.span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ accentColor: color }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function WhatIfPanel({ ehr = null }) {
  const [hr, setHr] = useState(72)
  const [hrv, setHrv] = useState(28)
  const [spo2, setSpo2] = useState(97)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef(null)

  const runSimulate = async (hrVal, hrvVal, spo2Val) => {
    setLoading(true)
    try {
      const { data } = await api.simulate({
        hr: hrVal, hrv: hrvVal, spo2: spo2Val,
        ...(ehr ? {
          age: ehr.age, bmi: ehr.bmi, sex: ehr.sex,
          hypertension: ehr.hypertension ? 1 : 0,
          sedative_use: 0, alcohol_use: 0,
          prior_ahi: ehr.prior_ahi ?? 10,
          ess_score: ehr.ess_score ?? 8,
          stopbang_score: ehr.stopbang_score ?? 3,
        } : {}),
      })
      setResult(data)
    } catch {
      // Heuristic fallback
      const riskVal = Math.max(0.02, Math.min(0.97,
        (100 - spo2Val) * 0.05 + Math.max(0, hrVal - 65) * 0.008 + Math.max(0, 30 - hrvVal) * 0.005
      ))
      setResult({ risk_score: riskVal, reasons: [], fallback: true })
    } finally {
      setLoading(false)
    }
  }

  // Debounced auto-simulate on slider change
  useEffect(() => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => runSimulate(hr, hrv, spo2), 350)
    return () => clearTimeout(debounceRef.current)
  }, [hr, hrv, spo2])

  const applyPreset = (p) => {
    setHr(p.hr)
    setHrv(p.hrv)
    setSpo2(p.spo2)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Sliders size={16} style={{ color: 'var(--primary)' }} />
        <h3 style={{ fontFamily: 'Space Grotesk', fontSize: '0.95rem' }}>What-If Simulator</h3>
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: -8 }}>
        Adjust sliders → calls real <code style={{ color: 'var(--primary)', fontSize: '0.72rem' }}>/simulate</code> endpoint live
      </p>

      {/* Presets */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            id={`preset-${p.label.replace(/\s+/g, '-').toLowerCase()}`}
            className="btn btn-ghost"
            style={{ fontSize: '0.72rem', padding: '5px 12px' }}
            onClick={() => applyPreset(p)}
          >
            {p.icon} {p.label}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <SliderRow
          id="sim-hr" label="Heart Rate"
          value={hr} min={40} max={130} step={1} unit="bpm"
          onChange={setHr} color="var(--accent-lavender)"
        />
        <SliderRow
          id="sim-hrv" label="HRV (RMSSD)"
          value={hrv} min={1} max={100} step={1} unit="ms"
          onChange={setHrv} color="var(--primary)"
        />
        <SliderRow
          id="sim-spo2" label="SpO₂"
          value={spo2} min={75} max={100} step={0.5} unit="%"
          onChange={setSpo2}
          color={spo2 < 90 ? 'var(--accent-coral)' : spo2 < 95 ? 'var(--accent-amber)' : 'var(--primary)'}
        />
      </div>

      {/* Result */}
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)', fontSize: '0.8rem' }}
          >
            <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} />
            Running simulation...
          </motion.div>
        ) : result ? (
          <motion.div
            key={result.risk_score}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
          >
            <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 16 }}>
              <RiskGauge score={result.risk_score} size={110} />
              <div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>30-min Event Risk</p>
                <p style={{ fontFamily: 'Space Grotesk', fontSize: '1.8rem', fontWeight: 700,
                  color: riskColor(result.risk_score), fontVariantNumeric: 'tabular-nums' }}>
                  {Math.round(result.risk_score * 100)}%
                </p>
                <p style={{ fontSize: '0.72rem', color: riskColor(result.risk_score), fontWeight: 500 }}>
                  {riskLabel(result.risk_score)} Risk
                </p>
                {result.fallback && (
                  <p style={{ fontSize: '0.65rem', color: 'var(--accent-amber)', marginTop: 4 }}>
                    heuristic — run training for real model
                  </p>
                )}
              </div>
            </div>

            {result.reasons?.length > 0 && (
              <div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                  Key Drivers (SHAP)
                </p>
                <ReasonCodes reasons={result.reasons.slice(0, 3)} />
              </div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  )
}
