// src/components/NightPlayback.jsx
// The demo-winning feature: scrubber + animated signal reveal + live predictions
// + real annotated event markers + "predicted N min early" badges.

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { Play, Pause, SkipBack, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { api, riskColor } from '../lib/api'
import RiskGauge from './RiskGauge'

const PLAYBACK_SPEEDS = [0.5, 1, 2, 4, 8]
const THRESHOLD = 0.55  // risk threshold for "predicted early" badge

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface-alt)',
      border: '1px solid var(--border)',
      borderRadius: 10,
      padding: '8px 12px',
      fontSize: '0.75rem',
    }}>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 4 }}>Min {label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color, margin: '2px 0' }}>
          {p.name}: <strong>{p.value?.toFixed(1) ?? '—'}</strong>
        </p>
      ))}
    </div>
  )
}

export default function NightPlayback({ subjectId, nightData, onMinuteChange, onPredictionChange, onPlayStateChange }) {
  const [currentMinute, setCurrentMinute] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(4)
  const [prediction, setPrediction] = useState(null)
  const [loadingPred, setLoadingPred] = useState(false)
  const [earlyBadges, setEarlyBadges] = useState({})  // eventStart → leadMin

  useEffect(() => {
    onPlayStateChange?.(isPlaying)
  }, [isPlaying, onPlayStateChange])

  const playRef = useRef(null)
  const totalMinutes = nightData?.signal?.length ?? 0
  const signal = nightData?.signal ?? []
  const events = nightData?.annotated_events ?? []

  // Throttled prediction fetch — don't hammer API during fast playback
  const predFetchTimer = useRef(null)
  const fetchPrediction = useCallback(async (minute) => {
    if (!subjectId) return
    clearTimeout(predFetchTimer.current)
    predFetchTimer.current = setTimeout(async () => {
      try {
        setLoadingPred(true)
        const { data } = await api.predictAt(subjectId, minute)
        setPrediction(data)
        onPredictionChange?.(data)

        // Check if we're predicting an event early
        if (data.risk_score >= THRESHOLD) {
          // Find the next real event after this minute
          const nextEvent = events.find(e => e.start_minute > minute && e.start_minute <= minute + 30)
          if (nextEvent) {
            const lead = nextEvent.start_minute - minute
            setEarlyBadges(prev => ({
              ...prev,
              [nextEvent.start_minute]: Math.min(prev[nextEvent.start_minute] ?? lead, lead),
            }))
          }
        }
      } catch {
        // Use heuristic fallback
        const row = signal.find(r => r.minute === minute) ?? signal[Math.min(minute, signal.length - 1)]
        const spo2 = row?.spo2 ?? 97
        const hr = row?.hr ?? 70
        setPrediction({
          risk_score: Math.max(0.05, Math.min(0.95, (100 - spo2) * 0.04 + Math.max(0, hr - 80) * 0.005)),
          reasons: [],
          minute,
          fallback: true,
        })
      } finally {
        setLoadingPred(false)
      }
    }, speed >= 4 ? 200 : 400)
  }, [subjectId, signal, events, speed])

  // Playback loop
  useEffect(() => {
    if (!isPlaying) {
      clearInterval(playRef.current)
      return
    }
    const intervalMs = 1000 / speed
    playRef.current = setInterval(() => {
      setCurrentMinute(prev => {
        const next = prev + 1
        if (next >= totalMinutes) {
          setIsPlaying(false)
          return prev
        }
        return next
      })
    }, intervalMs)
    return () => clearInterval(playRef.current)
  }, [isPlaying, speed, totalMinutes])

  // Fetch prediction whenever minute changes
  useEffect(() => {
    fetchPrediction(currentMinute)
    onMinuteChange?.(currentMinute)
  }, [currentMinute, fetchPrediction]) // eslint-disable-line react-hooks/exhaustive-deps

  // Always render the FULL signal array so the x-axis spans the whole night.
  // We reveal progress with a vertical "now" line + opacity fade.
  // visibleSignal was previously used but caused the "1-point at minute 0" bug
  // because the chart collapsed when only 1 data point existed.
  const fullSignal = signal  // entire night, always rendered

  const currentRow = signal[currentMinute] ?? {}
  const riskScore = prediction?.risk_score ?? 0

  // Debug: log length on every subject load (minute===0) so user can confirm fix
  if (process.env.NODE_ENV !== 'production') {
    if (fullSignal.length > 0 && currentMinute === 0) {
      // eslint-disable-next-line no-console
      console.info(
        `%c[NightPlayback] ${subjectId}: signal data array length = ${fullSignal.length} minutes`,
        'color:#5FD4C4;font-weight:bold'
      )
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ fontFamily: 'Space Grotesk', fontSize: '1.1rem', color: 'var(--text-primary)' }}>
            Night Playback
          </h2>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 2 }}>
            Real PhysioNet recording · Expert-annotated apnea events
          </p>
        </div>

        {/* Risk gauge (compact) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {prediction && (
            <AnimatePresence mode="wait">
              <motion.div
                key={Math.round(riskScore * 10)}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <RiskGauge score={riskScore} size={120} />
              </motion.div>
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Charts */}
      <div className="card" style={{ padding: 16, overflow: 'hidden' }}>
        {/* HR Chart — uses default category-mode XAxis (index-based) to guarantee
            all N minutes are rendered. type="number" with explicit domain caused
            Recharts to collapse to a single point on first render. */}
        <div style={{ marginBottom: 8 }}>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
            Heart Rate (bpm) &mdash; <strong style={{ color: 'var(--text-primary)' }}>{fullSignal.length}</strong> min recording
          </p>
          <ResponsiveContainer width="100%" height={90}>
            <LineChart data={fullSignal} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              {/* Category-based XAxis — renders every data point by array index.
                  tickFormatter maps index → actual minute value for display. */}
              <XAxis
                dataKey="minute"
                tick={{ fontSize: 9 }}
                tickCount={8}
                interval="preserveStartEnd"
                tickFormatter={(val) => val}
              />
              <YAxis domain={[40, 130]} tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              {events.map(ev => (
                <ReferenceLine key={ev.start_minute} x={ev.start_minute}
                  stroke="var(--accent-coral)" strokeDasharray="4 2" strokeWidth={1.5} opacity={0.6} />
              ))}
              {/* Current playhead position */}
              <ReferenceLine x={currentMinute} stroke="var(--primary)" strokeWidth={1.5} opacity={0.8} />
              <Line type="monotone" dataKey="hr" name="HR"
                stroke="var(--accent-lavender)" strokeWidth={1.5} dot={false}
                isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* HRV Chart */}
        <div style={{ marginBottom: 8 }}>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
            HRV RMSSD (ms)
          </p>
          <ResponsiveContainer width="100%" height={90}>
            <LineChart data={fullSignal} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="minute"
                tick={{ fontSize: 9 }}
                tickCount={8}
                interval="preserveStartEnd"
                tickFormatter={(val) => val}
              />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              {events.map(ev => (
                <ReferenceLine key={ev.start_minute} x={ev.start_minute}
                  stroke="var(--accent-coral)" strokeDasharray="4 2" strokeWidth={1.5} opacity={0.6} />
              ))}
              <ReferenceLine x={currentMinute} stroke="var(--primary)" strokeWidth={1.5} opacity={0.8} />
              <Line type="monotone" dataKey="hrv" name="HRV"
                stroke="var(--primary)" strokeWidth={1.5} dot={false}
                isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* SpO2 Chart */}
        <div>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
            SpO₂ (%)
          </p>
          <ResponsiveContainer width="100%" height={90}>
            <LineChart data={fullSignal} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="minute"
                tick={{ fontSize: 9 }}
                tickCount={8}
                interval="preserveStartEnd"
                tickFormatter={(val) => val}
              />
              <YAxis domain={[80, 100]} tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              {events.map(ev => (
                <ReferenceLine key={ev.start_minute} x={ev.start_minute}
                  stroke="var(--accent-coral)" strokeDasharray="4 2" strokeWidth={1.5} opacity={0.6} />
              ))}
              <ReferenceLine y={90} stroke="var(--accent-amber)" strokeDasharray="6 3"
                label={{ value: '90%', fill: 'var(--accent-amber)', fontSize: 9 }} />
              <ReferenceLine x={currentMinute} stroke="var(--primary)" strokeWidth={1.5} opacity={0.8} />
              <Line type="monotone" dataKey="spo2" name="SpO₂"
                stroke={riskColor(riskScore)} strokeWidth={1.5} dot={false}
                isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Timeline scrubber */}
      <div className="card" style={{ padding: 16 }}>
        {/* Scrubber bar with event markers */}
        <div style={{ position: 'relative', marginBottom: 12 }}>
          <input
            id="night-scrubber"
            type="range"
            min={0}
            max={Math.max(0, totalMinutes - 1)}
            value={currentMinute}
            onChange={(e) => {
              setIsPlaying(false)
              setCurrentMinute(Number(e.target.value))
            }}
            style={{ width: '100%' }}
          />
          {/* Event markers on scrubber */}
          {events.map(ev => {
            const pct = (ev.start_minute / (totalMinutes - 1)) * 100
            const lead = earlyBadges[ev.start_minute]
            return (
              <div key={ev.start_minute}>
                <div
                  title={`Real apnea event @ min ${ev.start_minute}`}
                  style={{
                    position: 'absolute',
                    left: `${pct}%`,
                    top: -8,
                    transform: 'translateX(-50%)',
                    width: 3,
                    height: 8,
                    background: 'var(--accent-coral)',
                    borderRadius: 1,
                  }}
                />
                {lead != null && currentMinute >= ev.start_minute - lead && currentMinute <= ev.start_minute + 2 && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{
                      position: 'absolute',
                      left: `${pct}%`,
                      top: -32,
                      transform: 'translateX(-50%)',
                    }}
                  >
                    <span className="lead-badge">
                      <CheckCircle2 size={10} />
                      {lead}m early
                    </span>
                  </motion.div>
                )}
              </div>
            )
          })}
        </div>

        {/* Controls row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <button
            id="playback-reset"
            className="btn btn-ghost"
            style={{ padding: '6px 10px' }}
            onClick={() => { setIsPlaying(false); setCurrentMinute(0); setPrediction(null) }}
          >
            <SkipBack size={14} />
          </button>

          <button
            id="playback-toggle"
            className="btn btn-primary"
            style={{ padding: '6px 14px' }}
            onClick={() => setIsPlaying(p => !p)}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          {/* Speed selector */}
          <div style={{ display: 'flex', gap: 4, marginLeft: 8 }}>
            {PLAYBACK_SPEEDS.map(s => (
              <button
                key={s}
                className={`btn ${speed === s ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                onClick={() => setSpeed(s)}
              >
                {s}×
              </button>
            ))}
          </div>

          {/* Current time */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Clock size={13} style={{ color: 'var(--text-secondary)' }} />
            <span style={{ fontSize: '0.82rem', fontVariantNumeric: 'tabular-nums', color: 'var(--text-secondary)' }}>
              {String(Math.floor(currentMinute / 60)).padStart(2, '0')}:
              {String(currentMinute % 60).padStart(2, '0')} / {' '}
              {String(Math.floor(totalMinutes / 60)).padStart(2, '0')}:
              {String(totalMinutes % 60).padStart(2, '0')}
            </span>
          </div>
        </div>

        {/* Current event status */}
        <AnimatePresence>
          {currentRow.apnea_label === 1 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{ marginTop: 10 }}
            >
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 14px', borderRadius: 10,
                background: 'var(--glow-coral)',
                border: '1px solid rgba(217,119,87,0.3)',
              }}>
                <AlertTriangle size={14} style={{ color: 'var(--accent-coral)', flexShrink: 0 }} />
                <span style={{ fontSize: '0.8rem', color: 'var(--accent-coral)', fontWeight: 500 }}>
                  ⚠ Real annotated apnea event (PhysioNet clinical label) — Minute {currentMinute}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Annotated events legend */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div style={{ width: 12, height: 2, background: 'var(--accent-coral)' }} />
          Real apnea events ({events.length} events, {nightData?.total_apnea_minutes} min total)
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div style={{ width: 10, height: 10, borderRadius: 999, background: 'rgba(111,227,208,0.2)', border: '1px solid var(--primary)' }} />
          Model threshold ({Math.round(THRESHOLD * 100)}%)
        </div>
        {prediction?.fallback && (
          <span style={{ color: 'var(--accent-amber)', fontSize: '0.7rem' }}>
            (heuristic — train model for real scores)
          </span>
        )}
      </div>
    </div>
  )
}
